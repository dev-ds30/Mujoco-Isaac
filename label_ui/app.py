
from flask import Flask, render_template, request, jsonify
import json, os, time
app=Flask(__name__)
@app.route('/')
def index():
    pairs_ui = os.path.join(os.path.dirname(__file__), 'pairs_ui.json')
    pairs = json.load(open(pairs_ui))['pairs'] if os.path.exists(pairs_ui) else []
    return render_template('index.html', pairs=pairs)
@app.route('/api/label', methods=['POST'])
def label():
    data=request.json
    out=os.path.join(os.path.dirname(__file__), 'labels.json')
    labels = json.load(open(out))['labels'] if os.path.exists(out) else []
    labels.append({'idx':data['idx'],'label':data['label'],'ts':int(time.time())})
    json.dump({'labels':labels}, open(out,'w'), indent=2)
    return jsonify(ok=True)
if __name__=='__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
