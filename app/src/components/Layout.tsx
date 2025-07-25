import Box from '@mui/material/Box';
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';

export default function Layout() {
  return (
    <Box
      id="Layout"
      sx={{
        display: 'flex',
        flexDirection: 'column',
        flexGrow: 1,
        alignItems: 'center',
        gap: 2,
        // padding: 0,
        margin: 0,
        justifyContent: 'center',
      }}
    >
      {/* Renders current route */}
      <Outlet />
      <Navbar />
    </Box>
  );
}
