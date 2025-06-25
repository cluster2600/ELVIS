import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';

const PerformancePage = () => {
  return (
    <Box>
      <Typography variant="h4" fontWeight="bold" gutterBottom>
        Performance Analytics
      </Typography>
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Performance Metrics
          </Typography>
          <Typography color="text.secondary">
            Detailed performance analytics and charts will be implemented here.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default PerformancePage;