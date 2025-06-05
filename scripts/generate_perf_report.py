#!/usr/bin/env python3
"""
Performance report generator for OpenRuntime Enhanced
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class PerformanceReporter:
    def __init__(self, results_dir: str = "."):
        self.results_dir = Path(results_dir)
        self.report_data = {}
    
    def load_results(self):
        """Load performance test results"""
        try:
            # Load stress test results
            if (self.results_dir / "perf_results.txt").exists():
                with open(self.results_dir / "perf_results.txt") as f:
                    stress_results = f.read()
                self.report_data["stress_test"] = self._parse_stress_results(stress_results)
            
            # Load GPU benchmark results
            if (self.results_dir / "gpu_benchmark.json").exists():
                with open(self.results_dir / "gpu_benchmark.json") as f:
                    gpu_results = json.load(f)
                self.report_data["gpu_benchmark"] = gpu_results
            
            # Load historical data if available
            self._load_historical_data()
            
        except Exception as e:
            print(f"Error loading results: {e}")
    
    def _parse_stress_results(self, results: str) -> Dict[str, Any]:
        """Parse stress test results from text output"""
        lines = results.split('\n')
        parsed = {
            "endpoints": {},
            "summary": {}
        }
        
        current_endpoint = None
        for line in lines:
            if "Testing" in line and "with" in line:
                current_endpoint = line.split()[1]
                parsed["endpoints"][current_endpoint] = {}
            elif "requests successful" in line:
                if current_endpoint:
                    success_rate = line.split()[1].split('/')[0]
                    parsed["endpoints"][current_endpoint]["successful"] = int(success_rate)
            elif "Average:" in line:
                if current_endpoint:
                    avg_time = float(line.split()[1].replace('s', ''))
                    parsed["endpoints"][current_endpoint]["avg_response_time"] = avg_time
            elif "Throughput:" in line:
                if current_endpoint:
                    throughput = float(line.split()[1])
                    parsed["endpoints"][current_endpoint]["throughput"] = throughput
        
        return parsed
    
    def _load_historical_data(self):
        """Load historical performance data for trend analysis"""
        history_file = self.results_dir / "performance_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.report_data["history"] = json.load(f)
            except:
                self.report_data["history"] = []
        else:
            self.report_data["history"] = []
    
    def generate_plots(self):
        """Generate performance visualization plots"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Response time comparison
        if "stress_test" in self.report_data:
            endpoints = list(self.report_data["stress_test"]["endpoints"].keys())
            response_times = [
                self.report_data["stress_test"]["endpoints"][ep].get("avg_response_time", 0)
                for ep in endpoints
            ]
            
            axes[0, 0].bar(endpoints, response_times)
            axes[0, 0].set_title('Average Response Time by Endpoint')
            axes[0, 0].set_ylabel('Response Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        if "stress_test" in self.report_data:
            throughputs = [
                self.report_data["stress_test"]["endpoints"][ep].get("throughput", 0)
                for ep in endpoints
            ]
            
            axes[0, 1].bar(endpoints, throughputs, color='green', alpha=0.7)
            axes[0, 1].set_title('Throughput by Endpoint')
            axes[0, 1].set_ylabel('Requests/Second')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # GPU benchmark results
        if "gpu_benchmark" in self.report_data:
            gpu_data = self.report_data["gpu_benchmark"]
            if "result" in gpu_data and "results" in gpu_data["result"]:
                results = gpu_data["result"]["results"]
                
                if "compute" in results:
                    compute_data = results["compute"]
                    sizes = [item["size"] for item in compute_data]
                    gflops = [item["gflops"] for item in compute_data]
                    
                    axes[1, 0].plot(sizes, gflops, marker='o', linewidth=2, markersize=8)
                    axes[1, 0].set_title('GPU Compute Performance (GFLOPS)')
                    axes[1, 0].set_xlabel('Matrix Size')
                    axes[1, 0].set_ylabel('GFLOPS')
                    axes[1, 0].grid(True, alpha=0.3)
        
        # Historical trend (if available)
        if "history" in self.report_data and self.report_data["history"]:
            history = self.report_data["history"]
            dates = [item["date"] for item in history]
            avg_response_times = [item.get("avg_response_time", 0) for item in history]
            
            axes[1, 1].plot(dates, avg_response_times, marker='o', linewidth=2)
            axes[1, 1].set_title('Performance Trend Over Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Average Response Time (s)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "performance_charts.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_html_report(self):
        """Generate comprehensive HTML performance report"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenRuntime Performance Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #007acc;
            padding-bottom: 20px;
        }}
        .metric {{
            background: #f8f9fa;
            border-left: 4px solid #007acc;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #007acc;
        }}
        .section {{
            margin: 30px 0;
        }}
        .charts {{
            text-align: center;
            margin: 30px 0;
        }}
        .charts img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #007acc;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-good {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-error {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>OpenRuntime Enhanced Performance Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>
        
        {self._generate_summary_section()}
        {self._generate_stress_test_section()}
        {self._generate_gpu_benchmark_section()}
        {self._generate_charts_section()}
        {self._generate_recommendations_section()}
    </div>
</body>
</html>
"""
        
        with open(self.results_dir / "performance_report.html", "w") as f:
            f.write(html_content)
    
    def _generate_summary_section(self) -> str:
        """Generate summary metrics section"""
        summary = "<div class='section'><h2>Summary Metrics</h2>"
        
        if "stress_test" in self.report_data:
            endpoints = self.report_data["stress_test"]["endpoints"]
            total_requests = sum(ep.get("successful", 0) for ep in endpoints.values())
            avg_response_time = sum(ep.get("avg_response_time", 0) for ep in endpoints.values()) / len(endpoints) if endpoints else 0
            avg_throughput = sum(ep.get("throughput", 0) for ep in endpoints.values()) / len(endpoints) if endpoints else 0
            
            summary += f"""
            <div class="metric">
                <strong>Total Successful Requests:</strong>
                <span class="metric-value">{total_requests:,}</span>
            </div>
            <div class="metric">
                <strong>Average Response Time:</strong>
                <span class="metric-value">{avg_response_time:.3f}s</span>
            </div>
            <div class="metric">
                <strong>Average Throughput:</strong>
                <span class="metric-value">{avg_throughput:.1f} req/s</span>
            </div>
            """
        
        summary += "</div>"
        return summary
    
    def _generate_stress_test_section(self) -> str:
        """Generate stress test results section"""
        if "stress_test" not in self.report_data:
            return ""
        
        section = "<div class='section'><h2>Stress Test Results</h2>"
        section += "<table><tr><th>Endpoint</th><th>Successful Requests</th><th>Avg Response Time</th><th>Throughput</th><th>Status</th></tr>"
        
        for endpoint, data in self.report_data["stress_test"]["endpoints"].items():
            successful = data.get("successful", 0)
            response_time = data.get("avg_response_time", 0)
            throughput = data.get("throughput", 0)
            
            # Determine status based on performance thresholds
            if response_time < 0.1 and throughput > 100:
                status = "<span class='status-good'>Excellent</span>"
            elif response_time < 0.5 and throughput > 50:
                status = "<span class='status-warning'>Good</span>"
            else:
                status = "<span class='status-error'>Needs Optimization</span>"
            
            section += f"""
            <tr>
                <td>{endpoint}</td>
                <td>{successful:,}</td>
                <td>{response_time:.3f}s</td>
                <td>{throughput:.1f} req/s</td>
                <td>{status}</td>
            </tr>
            """
        
        section += "</table></div>"
        return section
    
    def _generate_gpu_benchmark_section(self) -> str:
        """Generate GPU benchmark results section"""
        if "gpu_benchmark" not in self.report_data:
            return ""
        
        section = "<div class='section'><h2>GPU Benchmark Results</h2>"
        
        gpu_data = self.report_data["gpu_benchmark"]
        if "result" in gpu_data and "results" in gpu_data["result"]:
            results = gpu_data["result"]["results"]
            
            if "compute" in results:
                section += "<h3>Compute Performance</h3>"
                section += "<table><tr><th>Matrix Size</th><th>GFLOPS</th><th>Performance Level</th></tr>"
                
                for item in results["compute"]:
                    size = item["size"]
                    gflops = item["gflops"]
                    
                    if gflops > 1000:
                        perf_level = "<span class='status-good'>Excellent</span>"
                    elif gflops > 500:
                        perf_level = "<span class='status-warning'>Good</span>"
                    else:
                        perf_level = "<span class='status-error'>Below Average</span>"
                    
                    section += f"<tr><td>{size}</td><td>{gflops:.1f}</td><td>{perf_level}</td></tr>"
                
                section += "</table>"
            
            if "ml" in results:
                section += "<h3>ML Inference Performance</h3>"
                section += "<table><tr><th>Model</th><th>Latency (ms)</th><th>Throughput (FPS)</th><th>Performance</th></tr>"
                
                for item in results["ml"]:
                    model = item["model"]
                    latency = item["latency_ms"]
                    fps = item["throughput_fps"]
                    
                    if fps > 50:
                        perf_level = "<span class='status-good'>Excellent</span>"
                    elif fps > 20:
                        perf_level = "<span class='status-warning'>Good</span>"
                    else:
                        perf_level = "<span class='status-error'>Needs Optimization</span>"
                    
                    section += f"<tr><td>{model}</td><td>{latency:.1f}</td><td>{fps:.1f}</td><td>{perf_level}</td></tr>"
                
                section += "</table>"
        
        section += "</div>"
        return section
    
    def _generate_charts_section(self) -> str:
        """Generate charts section"""
        return """
        <div class='section'>
                            <h2>Performance Charts</h2>
            <div class='charts'>
                <img src='performance_charts.png' alt='Performance Charts' />
            </div>
        </div>
        """
    
    def _generate_recommendations_section(self) -> str:
        """Generate performance recommendations"""
        recommendations = []
        
        if "stress_test" in self.report_data:
            endpoints = self.report_data["stress_test"]["endpoints"]
            for endpoint, data in endpoints.items():
                response_time = data.get("avg_response_time", 0)
                throughput = data.get("throughput", 0)
                
                if response_time > 1.0:
                    recommendations.append(f"Optimize {endpoint}: High response time ({response_time:.2f}s)")
                
                if throughput < 10:
                    recommendations.append(f"Scale {endpoint}: Low throughput ({throughput:.1f} req/s)")
        
        if not recommendations:
            recommendations.append("All endpoints performing within acceptable parameters")
        
        section = "<div class='section'><h2>Recommendations</h2><ul>"
        for rec in recommendations:
            section += f"<li>{rec}</li>"
        section += "</ul></div>"
        
        return section
    
    def save_current_results(self):
        """Save current results to historical data"""
        if "stress_test" not in self.report_data:
            return
        
        current_data = {
            "date": datetime.now().isoformat(),
            "avg_response_time": sum(
                ep.get("avg_response_time", 0) 
                for ep in self.report_data["stress_test"]["endpoints"].values()
            ) / len(self.report_data["stress_test"]["endpoints"]),
            "avg_throughput": sum(
                ep.get("throughput", 0) 
                for ep in self.report_data["stress_test"]["endpoints"].values()
            ) / len(self.report_data["stress_test"]["endpoints"])
        }
        
        history = self.report_data.get("history", [])
        history.append(current_data)
        
        # Keep only last 30 entries
        if len(history) > 30:
            history = history[-30:]
        
        with open(self.results_dir / "performance_history.json", "w") as f:
            json.dump(history, f, indent=2)

def main():
    reporter = PerformanceReporter()
    reporter.load_results()
    reporter.generate_plots()
    reporter.generate_html_report()
    reporter.save_current_results()
    
    print("Performance report generated successfully!")
    print("Files created:")
    print("  - performance_report.html")
    print("  - performance_charts.png")
    print("  - performance_history.json")

if __name__ == "__main__":
    main()