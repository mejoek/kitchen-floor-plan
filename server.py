#!/usr/bin/env python3
"""
Kitchen Floor Plan Server
Serves static files and handles YAML save requests.
Works with all browsers (Safari, Firefox, Chrome, Edge).
"""

import http.server
import json
import os
import urllib.parse
from pathlib import Path

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class KitchenServer(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        """Handle save requests"""
        if self.path == '/save':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                filename = data.get('filename', 'kitchen.yaml')
                content = data.get('content', '')

                # Sanitize filename - only allow .yaml/.yml files in current directory
                filename = os.path.basename(filename)
                if not filename.endswith(('.yaml', '.yml')):
                    filename += '.yaml'

                filepath = os.path.join(DIRECTORY, filename)

                with open(filepath, 'w') as f:
                    f.write(content)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'success': True, 'filename': filename}).encode())
                print(f"Saved: {filename}")

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'error': str(e)}).encode())
                print(f"Save error: {e}")

        elif self.path == '/list-files':
            # List YAML files in directory
            try:
                files = [f for f in os.listdir(DIRECTORY) if f.endswith(('.yaml', '.yml'))]
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'files': sorted(files)}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

if __name__ == '__main__':
    os.chdir(DIRECTORY)
    with http.server.HTTPServer(('', PORT), KitchenServer) as httpd:
        print(f"Kitchen Floor Plan Server running at http://localhost:{PORT}")
        print(f"Serving files from: {DIRECTORY}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
