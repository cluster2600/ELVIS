import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Chip,
  IconButton,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AccountBalance,
  ShowChart,
  Refresh,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useSocket } from '../contexts/SocketContext';
import { apiService } from '../services/apiService';

const DashboardPage = () => {
  const [overview, setOverview] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { realTimeData, connected } = useSocket();

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError('');
      
      const [overviewRes, chartRes] = await Promise.all([
        apiService.getDashboardOverview(),
        apiService.getPerformanceChartData(7)
      ]);
      
      setOverview(overviewRes.data);
      setChartData(chartRes.data.data || []);
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const StatCard = ({ title, value, change, icon, color = 'primary' }) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box>
            <Typography color="text.secondary" gutterBottom variant="body2">
              {title}
            </Typography>
            <Typography variant="h4" component="div" fontWeight="bold">
              {value}
            </Typography>
            {change !== undefined && (
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                {change >= 0 ? (
                  <TrendingUp sx={{ color: 'success.main', mr: 0.5 }} />
                ) : (
                  <TrendingDown sx={{ color: 'error.main', mr: 0.5 }} />
                )}
                <Typography
                  variant="body2"
                  color={change >= 0 ? 'success.main' : 'error.main'}
                  fontWeight="medium"
                >
                  {change >= 0 ? '+' : ''}{change}%
                </Typography>
              </Box>
            )}
          </Box>
          <Box
            sx={{
              backgroundColor: `${color}.light`,
              borderRadius: '50%',
              p: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" action={
        <IconButton color="inherit" size="small" onClick={fetchDashboardData}>
          <Refresh />
        </IconButton>
      }>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">
          Dashboard Overview
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip
            label={connected ? 'Real-time' : 'Offline'}
            color={connected ? 'success' : 'error'}
            size="small"
          />
          <IconButton onClick={fetchDashboardData}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Statistics Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total P&L"
            value={`$${overview?.performance?.total_pnl?.toFixed(2) || '0.00'}`}
            change={5.2}
            icon={<AccountBalance sx={{ color: 'primary.main' }} />}
            color="primary"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Today's P&L"
            value={`$${overview?.performance?.today_pnl?.toFixed(2) || '0.00'}`}
            change={realTimeData.performance ? 2.1 : undefined}
            icon={<TrendingUp sx={{ color: 'success.main' }} />}
            color="success"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Trades"
            value={overview?.performance?.total_trades || 0}
            icon={<ShowChart sx={{ color: 'info.main' }} />}
            color="info"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Win Rate"
            value={`${((overview?.performance?.win_rate || 0) * 100).toFixed(1)}%`}
            change={1.5}
            icon={<TrendingUp sx={{ color: 'warning.main' }} />}
            color="warning"
          />
        </Grid>

        {/* Performance Chart */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Chart (7 Days)
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="cumulative_pnl"
                      stroke="#1976d2"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Bot Status */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Bot Status
              </Typography>
              {realTimeData.botStatus ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Chip
                    label={realTimeData.botStatus.running ? 'Running' : 'Stopped'}
                    color={realTimeData.botStatus.running ? 'success' : 'error'}
                  />
                  <Typography variant="body2">
                    Mode: {realTimeData.botStatus.mode}
                  </Typography>
                  <Typography variant="body2">
                    Strategy: {realTimeData.botStatus.strategy}
                  </Typography>
                  <Typography variant="body2">
                    Uptime: {realTimeData.botStatus.uptime}
                  </Typography>
                  <Typography variant="body2">
                    Health: {realTimeData.botStatus.health}
                  </Typography>
                </Box>
              ) : (
                <Typography color="text.secondary">
                  No real-time data available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Market Overview */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Market Overview
              </Typography>
              <Grid container spacing={2}>
                {Object.entries(realTimeData.marketData).map(([symbol, data]) => (
                  <Grid item xs={12} sm={6} md={4} key={symbol}>
                    <Box
                      sx={{
                        p: 2,
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                      }}
                    >
                      <Typography variant="subtitle1" fontWeight="bold">
                        {symbol}
                      </Typography>
                      <Typography variant="h6">
                        ${data.price?.toFixed(2)}
                      </Typography>
                      <Typography
                        variant="body2"
                        color={data.price_change_24h >= 0 ? 'success.main' : 'error.main'}
                      >
                        {data.price_change_24h >= 0 ? '+' : ''}{data.price_change_24h?.toFixed(2)}%
                      </Typography>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;