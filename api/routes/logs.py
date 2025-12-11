"""
============================================================
LOGS ROUTES - Access Logs & Statistics
============================================================
"""

from flask import Blueprint, request, jsonify

from ..database import get_logs, get_daily_stats

logs_bp = Blueprint('logs', __name__)


@logs_bp.route('/api/logs')
def access_logs():
    """Get access logs"""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        logs, total = get_logs(limit, offset)
        
        return jsonify({
            "success": True,
            "total": total,
            "limit": limit,
            "offset": offset,
            "logs": logs
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@logs_bp.route('/api/stats')
def stats():
    """Get daily statistics"""
    try:
        days = request.args.get('days', 7, type=int)
        
        daily_stats, overall = get_daily_stats(days)
        
        return jsonify({
            "success": True,
            "daily_stats": daily_stats,
            "overall": overall
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
