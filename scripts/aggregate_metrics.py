#!/usr/bin/env python3
"""
Script to aggregate metrics from processed videos and generate a professional HTML report.
"""

import os
import json
import argparse
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def format_duration(seconds):
    if not seconds:
        return "0s"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    return f"{int(m)}m {int(s)}s"

def generate_html_report(metrics, plots_info, report_path):
    # CSS Styling
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        body { 
            font-family: 'Inter', sans-serif; 
            line-height: 1.6; 
            color: #334155; 
            background-color: #f1f5f9; 
            margin: 0; 
            padding: 40px 20px; 
        }
        
        .container { 
            max-width: 1100px; 
            margin: 0 auto; 
            background: white; 
            padding: 40px; 
            border-radius: 12px; 
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); 
        }
        
        header { 
            text-align: center; 
            margin-bottom: 50px; 
            border-bottom: 2px solid #e2e8f0; 
            padding-bottom: 20px; 
        }
        
        h1 { 
            color: #1e293b; 
            margin-bottom: 10px; 
            font-size: 2.2em; 
        }
        
        .date { 
            color: #64748b; 
            font-size: 0.9em; 
        }
        
        .section-title { 
            color: #0f172a; 
            font-size: 1.5em; 
            margin-bottom: 25px; 
            display: flex; 
            align-items: center; 
            gap: 10px; 
        }
        
        .section-title::before {
            content: '';
            width: 6px;
            height: 24px;
            background: #3b82f6;
            border-radius: 4px;
            display: inline-block;
        }
        
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
            gap: 20px; 
            margin-bottom: 50px; 
        }
        
        .metric-card { 
            background: #f8fafc; 
            border: 1px solid #e2e8f0; 
            padding: 20px; 
            border-radius: 8px; 
            transition: transform 0.2s; 
        }
        
        .metric-card:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); 
        }
        
        .metric-label { 
            color: #64748b; 
            font-size: 0.85em; 
            text-transform: uppercase; 
            letter-spacing: 0.05em; 
            font-weight: 600; 
        }
        
        .metric-value { 
            color: #0f172a; 
            font-size: 1.8em; 
            font-weight: 700; 
            margin-top: 5px; 
        }
        
        .charts-container { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); 
            gap: 40px; 
        }
        
        .chart-box { 
            background: white; 
            border: 1px solid #e2e8f0; 
            border-radius: 8px; 
            padding: 20px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.05); 
        }
        
        .chart-box h3 { 
            margin-top: 0; 
            color: #334155; 
            font-size: 1.1em; 
            border-bottom: 1px solid #f1f5f9; 
            padding-bottom: 10px; 
            margin-bottom: 20px; 
        }
        
        img { 
            max-width: 100%; 
            height: auto; 
            border-radius: 4px; 
        }
        
        .explanation { 
            margin-top: 15px; 
            padding: 12px; 
            background: #f0f9ff; 
            border-radius: 6px; 
            color: #334155; 
            font-size: 0.9em; 
            border-left: 4px solid #3b82f6; 
        }
        
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 20px; 
        }
        
        th, td { 
            padding: 12px 15px; 
            text-align: left; 
            border-bottom: 1px solid #e2e8f0; 
        }
        
        th { 
            background-color: #f8fafc; 
            font-weight: 600; 
            color: #475569; 
        }
        
        .tag { 
            display: inline-block; 
            padding: 4px 10px; 
            border-radius: 20px; 
            background: #e2e8f0; 
            color: #475569; 
            font-size: 0.8em; 
            font-weight: 600; 
        }
        
        footer { 
            margin-top: 60px; 
            text-align: center; 
            color: #94a3b8; 
            font-size: 0.8em; 
        }
    </style>
    """
    
    # Content Logic
    total_videos = metrics['total_videos']
    formatted_duration = format_duration(metrics['total_duration'])
    avg_duration = format_duration(metrics['total_duration'] / total_videos) if total_videos else "0s"
    
    # LLM Stats
    avg_conf = 0
    if metrics['llm_stats']['avg_confidence']:
        avg_conf = np.mean(metrics['llm_stats']['avg_confidence']) * 100
        
    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reporte de Pipeline de Procesamiento</title>
        {css}
    </head>
    <body>
        <div class="container">
            <header>
                <h1>📊 Reporte de Pipeline</h1>
                <div class="date">Generado el {datetime.now().strftime('%d de %B de %Y a las %H:%M')}</div>
            </header>
            
            <!-- RESUMEN -->
            <div class="section">
                <h2 class="section-title">Resumen Ejecutivo</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Videos Procesados</div>
                        <div class="metric-value">{total_videos}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Videos Fallidos</div>
                        <div class="metric-value" style="color: #ef4444;">{metrics['status']['failed']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Duración Total</div>
                        <div class="metric-value">{formatted_duration}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Promedio LLM Confianza</div>
                        <div class="metric-value" style="color: #3b82f6;">{avg_conf:.1f}%</div>
                    </div>
                </div>
            </div>
            
            <!-- CHARTS -->
            <div class="section">
                <h2 class="section-title">Análisis Visual</h2>
                <div class="charts-container">
    """
    
    # Add charts
    charts_meta = {
        'categories': {
            'title': 'Distribución por Categoría',
            'desc': 'Cantidad de videos procesados agrupados por su temática. Muestra el balance del dataset.'
        },
        'speakers': {
            'title': 'Distribución de Hablantes',
            'desc': 'Frecuencia de número de hablantes por video. Indica la complejidad de diarización.'
        },
        'scatter': {
            'title': 'Rendimiento de Procesamiento',
            'desc': 'Relación entre duración del audio y tiempo de corrección LLM. Puntos oscuros indican menor confianza.'
        },
        'boxplot': {
            'title': 'Confianza por Categoría',
            'desc': 'Dispersión de la calidad de corrección (confianza) según el tema del video.'
        }
    }
    
    for key, filename in plots_info.items():
        meta = charts_meta.get(key, {'title': 'Gráfica', 'desc': 'Análisis de datos.'})
        html += f"""
                    <div class="chart-box">
                        <h3>{meta['title']}</h3>
                        <img src="{filename}" alt="{meta['title']}">
                        <div class="explanation">
                            <strong>Insight:</strong> {meta['desc']}
                        </div>
                    </div>
        """
        
    html += """
                </div>
            </div>
            
            <!-- DETALLES -->
            <div class="section">
                <h2 class="section-title" style="margin-top: 50px;">Detalles por Categoría</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Categoría</th>
                            <th>Cantidad</th>
                            <th>Porcentaje</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Table rows
    total_cats = sum(metrics['categories'].values())
    sorted_cats = sorted(metrics['categories'].items(), key=lambda x: x[1], reverse=True)
    
    for cat, count in sorted_cats:
        pct = (count / total_cats * 100) if total_cats else 0
        html += f"""
                        <tr>
                            <td><span class="tag">{cat}</span></td>
                            <td>{count}</td>
                            <td>{pct:.1f}%</td>
                        </tr>
        """
        
    html += """
                    </tbody>
                </table>
            </div>
            
            <footer>
                fromPodtoCast Pipeline Analytics &bull; Automated Report
            </footer>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✅ HTML Report saved to: {report_path}")

def generate_plots(metrics, output_dir):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns # Optional styling
        sns.set_theme(style="whitegrid")
    except ImportError:
        print("Matplotlib not installed. Skipping plots.")
        return {}

    print(f"\nGenerando gráficas en {output_dir}...")
    plots_info = {}
    
    # 1. Categories
    plt.figure(figsize=(8, 5))
    cats = list(metrics['categories'].keys())
    counts = list(metrics['categories'].values())
    sns.barplot(x=cats, y=counts, palette="viridis")
    plt.title('')
    plt.xlabel('')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    img_name = 'metrics_categories.png'
    plt.savefig(output_dir / img_name, dpi=100)
    plt.close()
    plots_info['categories'] = img_name

    # 2. Speakers
    if metrics['diarization_stats']['unique_speakers']:
        plt.figure(figsize=(8, 5))
        sns.histplot(metrics['diarization_stats']['unique_speakers'], discrete=True, color="skyblue")
        plt.title('')
        plt.xlabel('Número de Hablantes')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        img_name = 'metrics_speakers.png'
        plt.savefig(output_dir / img_name, dpi=100)
        plt.close()
        plots_info['speakers'] = img_name

    # 3. Scatter
    scatter_data = metrics.get('scatter_data', [])
    if scatter_data:
        plt.figure(figsize=(8, 5))
        durs = [x['duration'] / 60 for x in scatter_data] 
        times = [x['time'] / 60 for x in scatter_data]    
        confs = [x['conf'] for x in scatter_data] 
        
        sc = plt.scatter(durs, times, c=confs, cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(sc, label='Confianza')
        plt.xlabel('Duración Audio (min)')
        plt.ylabel('Tiempo Proceso (min)')
        plt.tight_layout()
        img_name = 'metrics_duration_vs_time.png'
        plt.savefig(output_dir / img_name, dpi=100)
        plt.close()
        plots_info['scatter'] = img_name

    # 4. Boxplot
    if scatter_data:
        cat_confs = defaultdict(list)
        for x in scatter_data:
            cat_confs[x['cat']].append(x['conf'])
        
        if cat_confs:
            plt.figure(figsize=(10, 6))
            labels = []
            values = []
            for cat, c_list in cat_confs.items():
                labels.append(cat)
                values.append(c_list)
            
            plt.boxplot(values, labels=labels)
            plt.ylabel('Score Confianza')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            img_name = 'metrics_confidence_boxplot.png'
            plt.savefig(output_dir / img_name, dpi=100)
            plt.close()
            plots_info['boxplot'] = img_name
            
    return plots_info

def main():
    parser = argparse.ArgumentParser(description="Aggregate pipeline metrics")
    parser.add_argument(
        '--data-path', 
        default='/media/ttech-main/8464b39b-ba5b-49c5-9e05-010122b9874d',
        help="Path to the data directory containing processed_videos.json and logs/"
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    registry_path = data_path / 'processed_videos.json'
    logs_dir = data_path / 'logs'

    if not registry_path.exists():
        print(f"Registry not found at {registry_path}")
        return

    print(f"Loading registry from {registry_path}...")
    registry = load_json(registry_path)
    if not registry:
        return

    print(f"Loading logs from {logs_dir}...")
    audio_to_log = {}
    log_files = list(logs_dir.glob('*.log'))
    print(f"Found {len(log_files)} log files.")
    
    for log_file in log_files:
        data = load_json(log_file)
        if data and 'input_audio' in data:
            input_path = os.path.normpath(data['input_audio'])
            audio_to_log[input_path] = data
            fname = os.path.basename(input_path)
            if fname not in audio_to_log:
                audio_to_log[fname] = data

    metrics = {
        'total_videos': 0,
        'total_duration': 0.0,
        'total_segments_registry': 0,
        'categories': defaultdict(int),
        'llm_stats': {
            'avg_confidence': [],
            'processing_time': [],
            'total_changes': 0,
            'corrected_segments': 0
        },
        'diarization_stats': {
            'unique_speakers': []
        },
        'status': {
            'success': 0,
            'failed': 0,
            'skipped': 0
        },
        'scatter_data': []
    }

    processed = registry.get('processed', {})
    failed = registry.get('failed', {})
    metrics['status']['success'] = len(processed)
    metrics['status']['failed'] = len(failed)
    
    for vid_id, info in processed.items():
        metrics['total_videos'] += 1
        dur = info.get('duration', 0)
        metrics['total_duration'] += dur
        cat = info.get('category', 'unknown')
        metrics['categories'][cat] += 1
        
        seg_str = info.get('segments', '0')
        try:
            seg_count = int(seg_str.split()[0])
        except:
            seg_count = 0
        metrics['total_segments_registry'] += seg_count
        
        audio_path_full = info.get('audio_path', '')
        if not audio_path_full:
            continue
            
        audio_path = os.path.normpath(audio_path_full)
        log_data = audio_to_log.get(audio_path)
        
        if not log_data:
            fname = os.path.basename(audio_path)
            log_data = audio_to_log.get(fname)
        
        if log_data:
            llm = log_data.get('llm_correction', {})
            if llm.get('enabled'):
                if 'avg_confidence' in llm:
                    conf = llm['avg_confidence']
                    metrics['llm_stats']['avg_confidence'].append(conf)
                
                if 'processing_time' in llm:
                    proc_time = llm['processing_time']
                    metrics['llm_stats']['processing_time'].append(proc_time)
                    if dur > 0 and proc_time > 0:
                        metrics['scatter_data'].append({
                            'duration': dur,
                            'time': proc_time,
                            'cat': cat,
                            'conf': conf if 'avg_confidence' in llm else 0
                        })

                metrics['llm_stats']['total_changes'] += llm.get('total_changes', 0)
                metrics['llm_stats']['corrected_segments'] += llm.get('corrected', 0)
            
            diar = log_data.get('diarization', {})
            spk_count = diar.get('unique_speakers', 0)
            if spk_count > 0:
                metrics['diarization_stats']['unique_speakers'].append(spk_count)

    plots_info = generate_plots(metrics, data_path)
    
    report_path = data_path / 'pipeline_report.html'
    generate_html_report(metrics, plots_info, report_path)

if __name__ == "__main__":
    main()
