import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';

const TradingPage = () => {
  return (
    <Box>
      <Typography variant="h4" fontWeight="bold" gutterBottom>
        Trading
      </Typography>
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Trading Controls
          </Typography>
          <Typography color="text.secondary">
            Trading controls and live market data will be implemented here.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default TradingPage;