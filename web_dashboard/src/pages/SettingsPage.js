import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';

const SettingsPage = () => {
  return (
    <Box>
      <Typography variant="h4" fontWeight="bold" gutterBottom>
        Settings
      </Typography>
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Bot Configuration
          </Typography>
          <Typography color="text.secondary">
            Bot settings and configuration options will be implemented here.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default SettingsPage;