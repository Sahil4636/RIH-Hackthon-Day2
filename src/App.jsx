import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Switch, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import UploadAnalyze from './components/UploadAnalyze';
import ShelfReport from './components/ShelfReport';
import Alerts from './components/Alerts';

const App = () => {
  const [image, setImage] = useState(null);
  const [analysisResults, setAnalysisResults] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [dashboardData, setDashboardData] = useState({});

  return (
    <Router>
      <nav>
        <ul>
          <li><Link to="/">Dashboard</Link></li>
          <li><Link to="/upload-analyze">Upload/Analyze</Link></li>
          <li><Link to="/shelf-report">Shelf Report</Link></li>
          <li><Link to="/alerts">Alerts</Link></li>
        </ul>
      </nav>
      
      <Switch>
        <Route path="/" exact>
          <Dashboard data={dashboardData} />
        </Route>
        <Route path="/upload-analyze">
          <UploadAnalyze 
            setImage={setImage} 
            setAnalysisResults={setAnalysisResults} 
          />
        </Route>
        <Route path="/shelf-report">
          <ShelfReport analysisResults={analysisResults} />
        </Route>
        <Route path="/alerts">
          <Alerts alerts={alerts} />
        </Route>
      </Switch>
    </Router>
  );
};

export default App;
