import Box from '@mui/material/Box';
import { type Theme, useTheme } from '@mui/material/styles';
import * as d3 from 'd3';
import moment from 'moment-timezone';
import type React from 'react';
import { useEffect, useRef } from 'react';
import type { SleepRecord } from '../../../server/src/db/sleepRecordsSchema.ts';

interface SleepBarChartProps {
  width?: number;
  height?: number;
  sleepRecords?: SleepRecord[];
  selectedSleepRecord?: SleepRecord;
  setSelectedSleepRecord: React.Dispatch<
    React.SetStateAction<SleepRecord | undefined>
  >;
}

interface AxisProps {
  chartG: d3.Selection<SVGGElement, unknown, null, undefined>;
  xScale: d3.ScaleBand<string>;
  yScale: d3.ScaleLinear<number, number>;
  height: number;
  theme: Theme;
  selectedSleepRecord: SleepRecord;
}

// ----------------------------------------------------------------------
// Convert UTC string -> local Moment
// ----------------------------------------------------------------------
function convertToLocalTime(date: string): moment.Moment {
  return moment(date).local();
}

// ----------------------------------------------------------------------
// Convert local date -> fractional hour [0..24), e.g. 1:30 AM => 1.5
// ----------------------------------------------------------------------
function dateToHourOfDay(dateString: string): number {
  const date = convertToLocalTime(dateString);
  return date.hours() + date.minutes() / 60;
}

// ----------------------------------------------------------------------
// SHIFT HOUR if < 18 => hour + 24 (so "below 6PM" is shown as next day).
// Example: 3 AM => 3 (which is < 18) => 3+24=27.
//  9 PM => 21 => no shift.
// ----------------------------------------------------------------------
function shiftHour(hour: number): number {
  return hour < 18 ? hour + 24 : hour;
}

// ----------------------------------------------------------------------
// Format the shifted hour in "ha" (12-hour clock) style.
//   - If hour >= 24, subtract 24 to get the "next day" hour for labeling
//   - e.g. 27 => 3 AM
//   - e.g. 18 => 6 PM
// ----------------------------------------------------------------------
function formatShiftedHour(hour: number): string {
  // If hour >= 24, bring it back into [0..24).
  let base = hour;
  while (base >= 24) {
    base -= 24;
  }
  // Now base is e.g. 3 => "3 AM", 6 => "6 PM", etc.
  return moment().startOf('day').add(base, 'hours').format('ha').toUpperCase();
}

// ----------------------------------------------------------------------
// Format X-axis ticks => "MON", "TUE", etc
// ----------------------------------------------------------------------
function formatDayLabel(dateString: string): string {
  const localTime = moment(dateString).local();
  if (localTime.hour() < 6) {
    return localTime.subtract(1, 'day').format('ddd').toUpperCase();
  } else {
    return localTime.format('ddd').toUpperCase();
  }
}

// ----------------------------------------------------------------------
// Create X and Y scales, but Y domain is dynamic based on actual sleep times
// ----------------------------------------------------------------------
function createScales({
  data,
  width,
  height,
}: {
  data: SleepRecord[];
  width: number;
  height: number;
}) {
  // X scale: one band per "entered_bed_at"
  const xScale = d3
    .scaleBand<string>()
    .domain(data.map((d) => d.entered_bed_at))
    .range([0, width])
    .padding(0.5);

  // 1) Collect all shifted hours from data
  const allShiftedHours: number[] = [];
  data.forEach((rec) => {
    rec.present_intervals.forEach(([startStr, endStr]) => {
      const startHour = dateToHourOfDay(startStr);
      const endHour = dateToHourOfDay(endStr);

      // Include unshifted values to prevent min from shifting incorrectly
      allShiftedHours.push(shiftHour(startHour), shiftHour(endHour));
    });
  });

  if (allShiftedHours.length === 0) {
    // Fallback: show 6 PM -> 6 PM
    return {
      xScale,
      yScale: d3.scaleLinear().domain([18, 42]).range([0, height]),
    };
  }

  // 2) Compute min/max from raw times
  const rawMin = Math.min(...allShiftedHours);
  const rawMax = Math.max(...allShiftedHours);

  // 3) Adjust min so pre-midnight times stay in proper range
  const yMin = Math.floor(rawMin) - 1;
  const yMax = Math.ceil(rawMax) + 1;

  // 4) Build the Y scale from [yMin..yMax]
  const yScale = d3.scaleLinear().domain([yMin, yMax]).range([0, height]);

  return { xScale, yScale };
}

// ----------------------------------------------------------------------
// Draw axes with grid lines
// ----------------------------------------------------------------------
function drawAxes({
  chartG,
  xScale,
  yScale,
  height,
  theme,
  selectedSleepRecord,
}: AxisProps) {
  // X-axis
  const xAxis = d3.axisBottom<string>(xScale).tickFormat(formatDayLabel);
  const xAxisGroup = chartG
    .append('g')
    .attr('class', 'x-axis')
    .attr('transform', `translate(0, ${height})`)
    .call(xAxis);

  // .attr('fill', theme.palette.grey[500])
  xAxisGroup
    .selectAll('text')
    .attr(
      'fill',
      (d) =>
        selectedSleepRecord && d === selectedSleepRecord.entered_bed_at
          ? theme.palette.grey[100] // Highlight selected bar's tick
          : theme.palette.grey[500], // Default white for others
    )
    // @ts-ignore
    .style('font-family', theme.typography.body1.fontFamily);
  xAxisGroup.selectAll('.tick line').remove();
  xAxisGroup.select('.domain').remove();

  // Y-axis
  const yAxis = d3
    .axisLeft<number>(yScale)
    .ticks(8) // adjust as needed
    .tickFormat((d) => formatShiftedHour(d));

  const yAxisGroup = chartG.append('g').attr('class', 'y-axis').call(yAxis);

  yAxisGroup
    .selectAll('text')
    .attr('fill', theme.palette.grey[500])
    // @ts-ignore
    .style('font-family', theme.typography.body1.fontFamily);
  yAxisGroup.selectAll('.tick line').remove();
  yAxisGroup.select('.domain').remove();

  // Horizontal grid lines
  chartG
    .selectAll('.grid-line')
    .data(yScale.ticks(8))
    .enter()
    .append('line')
    .attr('class', 'grid-line')
    .attr('x1', 0)
    .attr('x2', xScale.range()[1])
    .attr('y1', (d) => yScale(d))
    .attr('y2', (d) => yScale(d))
    .attr('stroke', theme.palette.divider);
}

// ----------------------------------------------------------------------
// Plot each record by shifting its start/end times & mapping to Y
// ----------------------------------------------------------------------
function plotSleepRecords({
  chartG,
  data,
  xScale,
  yScale,
  theme,
  selectedSleepRecord,
  setSelectedSleepRecord,
}: {
  chartG: d3.Selection<SVGGElement, unknown, null, undefined>;
  data: SleepRecord[];
  xScale: d3.ScaleBand<string>;
  yScale: d3.ScaleLinear<number, number>;
  theme: Theme;
  selectedSleepRecord: SleepBarChartProps['selectedSleepRecord'];
  setSelectedSleepRecord: SleepBarChartProps['setSelectedSleepRecord'];
}) {
  // Clear existing
  chartG.selectAll('.day-group').remove();

  chartG
    .selectAll('.day-group')
    .data(data)
    .enter()
    .append('g')
    .attr('class', 'day-group')
    .each(function (sleepRecord) {
      const gDay = d3.select(this);
      const rectX = xScale(sleepRecord.entered_bed_at) ?? 0;
      const rectWidth = xScale.bandwidth();
      const isSelected = sleepRecord.id === selectedSleepRecord?.id;

      sleepRecord.present_intervals.forEach(([startStr, endStr], i) => {
        if (startStr < sleepRecord.entered_bed_at) {
          if (endStr > sleepRecord.entered_bed_at) {
            startStr = sleepRecord.entered_bed_at;
          } else {
            return;
          }
        }

        if (endStr > sleepRecord.left_bed_at) {
          if (startStr < sleepRecord.left_bed_at) {
            endStr = sleepRecord.left_bed_at;
          } else {
            return;
          }
        }

        // Shift hours for each interval
        const s = shiftHour(dateToHourOfDay(startStr));
        const e = shiftHour(dateToHourOfDay(endStr));

        const y1 = yScale(s);
        const y2 = yScale(e);

        gDay
          .append('rect')
          .attr('x', rectX)
          .attr('y', Math.min(y1, y2)) // in case reversed
          .attr('width', rectWidth)
          .attr('height', Math.abs(y2 - y1)) // in case reversed
          .attr(
            'fill',
            isSelected ? theme.palette.grey[100] : theme.palette.grey[900],
          )
          .attr('rx', 2)
          .attr('class', 'bar')
          .attr('data-id', `${sleepRecord.entered_bed_at}-${i}`)
          .on('click', () => setSelectedSleepRecord(sleepRecord));
        if (isSelected) {
          gDay
            .append('text')
            .attr('x', rectX + rectWidth / 2)
            .attr('y', yScale.range()[1] + 40) // Placing arrow just below the X-axis
            .attr('text-anchor', 'middle')
            .attr('fill', theme.palette.grey[100])
            .attr('font-size', '16px')
            .text('▲'); // Unicode up arrow
        }
      });
    });
}

// ----------------------------------------------------------------------
// Main React Component
// ----------------------------------------------------------------------
export default function SleepBarChart({
  width = 400,
  height = 400,
  sleepRecords,
  selectedSleepRecord,
  setSelectedSleepRecord,
}: SleepBarChartProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const theme = useTheme();

  useEffect(() => {
    if (!sleepRecords?.length) return;

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Prep the SVG
    const svg = d3
      .select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .style('background', theme.palette.background.default)
      // @ts-ignore
      .style('font-family', theme.typography.body1.fontFamily);

    // Clear old content
    svg.selectAll('*').remove();

    // Create main group
    const chartG = svg
      .append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Create scales (now dynamic on min/max shifted hour)
    const { xScale, yScale } = createScales({
      data: sleepRecords,
      width: innerWidth,
      height: innerHeight,
    });
    if (selectedSleepRecord === undefined) return;
    // Draw axes
    drawAxes({
      chartG,
      xScale,
      yScale,
      height: innerHeight,
      theme,
      selectedSleepRecord,
    });

    // Plot the sleep intervals
    plotSleepRecords({
      chartG,
      data: sleepRecords,
      xScale,
      yScale,
      theme,
      selectedSleepRecord,
      setSelectedSleepRecord,
    });
  }, [
    sleepRecords,
    width,
    height,
    theme,
    selectedSleepRecord,
    setSelectedSleepRecord,
  ]);
  if (sleepRecords?.length === 0) return null;
  return (
    <Box sx={{ width: width }}>
      <svg ref={svgRef} />
    </Box>
  );
}
