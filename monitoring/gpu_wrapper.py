from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse the query string
        command = "gpustat --json --no-header"

        try:
            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Send response status code
            self.send_response(200)
            # Send headers
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            # Write the output to the response
            self.wfile.write(result.stdout.encode())
        except subprocess.CalledProcessError as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(e.stderr.encode())


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
