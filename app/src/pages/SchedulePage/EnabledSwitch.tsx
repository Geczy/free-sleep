import { FormControlLabel, Switch } from '@mui/material';
import { useAppStore } from '@state/appStore.tsx';
import { useScheduleStore } from './scheduleStore.tsx';

export default function EnabledSwitch() {
  const { isUpdating } = useAppStore();
  const { selectedSchedule, updateSelectedSchedule } = useScheduleStore();

  return (
    <FormControlLabel
      control={
        <Switch
          checked={selectedSchedule?.power.enabled || false}
          onChange={() => {
            updateSelectedSchedule({
              power: {
                enabled: !selectedSchedule?.power.enabled,
              },
            });
          }}
          disabled={isUpdating}
        />
      }
      label={selectedSchedule?.power.enabled ? 'Enabled' : 'Disabled'}
    />
  );
}
