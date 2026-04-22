import React from 'react';
import PropTypes from 'prop-types';
import './KPICard.css'; // Assuming you have a CSS file for styling

const KPICard = ({ icon, title, value, trend }) => {
    return (
        <div className="kpi-card">
            <div className="kpi-icon">{icon}</div>
            <h3 className="kpi-title">{title}</h3>
            <p className="kpi-value">{value}</p>
            <p className={`kpi-trend ${trend >= 0 ? 'up' : 'down'}`}>{trend >= 0 ? '+' : ''}{trend}%</p>
        </div>
    );
};

KPICard.propTypes = {
    icon: PropTypes.node.isRequired,
    title: PropTypes.string.isRequired,
    value: PropTypes.oneOfType([
        PropTypes.string,
        PropTypes.number
    ]).isRequired,
    trend: PropTypes.number.isRequired,
};

export default KPICard;