#!/usr/bin/env python3
"""
Simple Heartbeat Manager Starter
===============================
Simplified command to start the heartbeat monitoring system.
"""

import sys
import time
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from monitoring.heartbeat_manager import start_heartbeat_manager
except ImportError as e:
    print(f"❌ Error importing heartbeat manager: {e}")
    print("Make sure you're in the NetworkFed directory and monitoring module exists.")
    sys.exit(1)


def main():
    """Start heartbeat manager with simple logging."""
    print("🔊 Starting Heartbeat Manager...")
    print("=" * 50)
    print("📡 This will monitor datasite availability on port 8888")
    print("🏭 Factory datasites should send heartbeats to this endpoint")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Start the heartbeat manager
        manager = start_heartbeat_manager()
        
        if manager is None:
            print("❌ Failed to start heartbeat manager")
            return 1
            
        print("✅ Heartbeat manager started successfully!")
        print("📊 Monitoring datasite heartbeats on http://localhost:8888")
        print("🔍 Available endpoints:")
        print("   - POST /heartbeat          : Receive heartbeat from datasites")
        print("   - GET  /status/<site_id>   : Check specific datasite status")
        print("   - GET  /status/all         : Check all datasite statuses")
        print("   - GET  /available          : Get list of available datasites")
        print("\n⏳ Waiting for datasite heartbeats...")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(10)
                # Show periodic status update
                status = manager.get_all_datasite_status()
                if status:
                    available_count = len([s for s in status.values() if s.get('status') == 'online'])
                    total_count = len(status)
                    print(f"📈 Status update: {available_count}/{total_count} datasites available")
                    
                    # Show detailed status for each datasite
                    if available_count > 0:
                        online_sites = [name for name, info in status.items() if info.get('status') == 'online']
                        print(f"✅ Online datasites: {', '.join(online_sites)}")
                else:
                    print("🔄 No datasite heartbeats received yet...")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Heartbeat manager stopped by user")
            return 0
            
    except Exception as e:
        print(f"❌ Error starting heartbeat manager: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
