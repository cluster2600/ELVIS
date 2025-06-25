# ELVIS Trading Bot - Web Dashboard

A modern, real-time web dashboard for monitoring and controlling the ELVIS Trading Bot built with React and Material-UI.

## Features

- **Real-time Monitoring**: Live updates via WebSocket connections
- **Trading Controls**: Start/stop bot, configure strategies
- **Performance Analytics**: Interactive charts and performance metrics
- **Market Data**: Live price feeds and technical indicators
- **Alert System**: Real-time notifications and alerts
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Technology Stack

- **Frontend**: React 18, Material-UI 5, Recharts
- **Real-time**: Socket.IO client
- **HTTP Client**: Axios
- **Build Tool**: Create React App

## Quick Start

### Prerequisites

- Node.js 16+ and npm
- ELVIS Trading Bot API server running on port 5000

### Installation

1. Navigate to the dashboard directory:
```bash
cd web_dashboard
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The dashboard will open at `http://localhost:3000`

### Default Login

- **Username**: admin
- **Password**: admin

## Project Structure

```
web_dashboard/
├── public/
│   ├── index.html          # Main HTML template
│   └── manifest.json       # PWA manifest
├── src/
│   ├── components/
│   │   ├── common/          # Reusable components
│   │   └── layout/          # Layout components
│   ├── contexts/
│   │   ├── AuthContext.js   # Authentication context
│   │   └── SocketContext.js # WebSocket context
│   ├── pages/
│   │   ├── DashboardPage.js # Main dashboard
│   │   ├── TradingPage.js   # Trading controls
│   │   ├── PerformancePage.js # Analytics
│   │   └── SettingsPage.js  # Configuration
│   ├── services/
│   │   └── apiService.js    # API client
│   ├── App.js               # Main app component
│   └── index.js             # Entry point
└── package.json
```

## Environment Variables

Create a `.env` file in the web_dashboard directory:

```env
REACT_APP_API_URL=http://localhost:5000
```

## Features Overview

### Dashboard
- Bot status and health monitoring
- Real-time performance metrics
- Market data overview
- Performance charts

### Real-time Updates
- WebSocket connection to trading bot
- Live price feeds
- Trade execution notifications
- Bot status changes

### Authentication
- JWT-based authentication
- Secure token storage
- Automatic token refresh

## API Integration

The dashboard integrates with the ELVIS Trading Bot API:

- **Authentication**: POST `/api/auth/login`
- **Bot Control**: POST `/api/bot/start`, `/api/bot/stop`
- **Dashboard Data**: GET `/api/dashboard/overview`
- **Real-time**: WebSocket connection for live updates

## Development

### Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

### Building for Production

```bash
npm run build
```

This creates a `build` directory with optimized production files.

### Deployment

The dashboard can be deployed to any static hosting service:

1. Build the project: `npm run build`
2. Deploy the `build` directory contents
3. Configure your web server to serve the API from the same domain or configure CORS

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the ELVIS Trading Bot system.