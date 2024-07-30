from flask import Flask, render_template, request, jsonify, Response
import pubmed_scraper
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/scrape', methods=['GET'])
def scrape():
    max_results = int(request.args.get('max_results', 50))
    protein_target = request.args.get('protein_target', '')
    max_publications = request.args.get('max_publications')
    max_publications = int(max_publications) if max_publications else None
    
    query = pubmed_scraper.generate_query(protein_target)
    
    def generate():
        results_count = 0
        for article in pubmed_scraper.scrape_pubmed(query, protein_target, max_results=max_results, max_publications=max_publications):
            results_count += 1
            progress = min(results_count / max_results * 100, 100)
            yield f"data: {json.dumps({'progress': progress, 'article': article})}\n\n"
        
        if results_count < max_results:
            yield f"data: {json.dumps({'progress': 100, 'message': f'Search completed. Found {results_count} results.'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)