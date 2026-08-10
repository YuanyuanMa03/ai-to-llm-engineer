import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.4/full/pyodide.mjs';

let pyodidePromise;

function postStatus(id, stage) {
  self.postMessage({ type: 'status', id, stage });
}

async function getPyodide(id) {
  if (!pyodidePromise) {
    postStatus(id, 'loading-runtime');
    pyodidePromise = loadPyodide().catch((error) => {
      pyodidePromise = undefined;
      throw error;
    });
  }
  return pyodidePromise;
}

async function captureFigures(pyodide, globals) {
  const json = await pyodide.runPythonAsync(`
import sys as _sys
import json as _json

_lab_images = []
if "matplotlib.pyplot" in _sys.modules:
    import io as _io
    import base64 as _base64
    import matplotlib.pyplot as _plt
    for _number in _plt.get_fignums():
        _figure = _plt.figure(_number)
        _buffer = _io.BytesIO()
        _figure.savefig(_buffer, format="png", dpi=120, bbox_inches="tight")
        _lab_images.append(_base64.b64encode(_buffer.getvalue()).decode("ascii"))
        _buffer.close()
    _plt.close("all")
_json.dumps(_lab_images)
`, { globals });
  return JSON.parse(String(json || '[]'));
}

function serialiseResult(value) {
  if (value === undefined || value === null) return '';
  try {
    return String(value);
  } finally {
    if (value && typeof value.destroy === 'function') value.destroy();
  }
}

self.onmessage = async function (event) {
  const { id, cells, targetIndex } = event.data || {};
  if (!id || !Array.isArray(cells) || !cells.length) return;

  let globals;
  try {
    const pyodide = await getPyodide(id);
    postStatus(id, 'loading-packages');
    await pyodide.loadPackagesFromImports(cells.join('\n\n'));

    const stdout = [];
    const stderr = [];
    pyodide.setStdout({ batched: (line) => stdout.push(line) });
    pyodide.setStderr({ batched: (line) => stderr.push(line) });
    pyodide.setStdin({ error: true });

    const dict = pyodide.globals.get('dict');
    globals = dict();
    dict.destroy();

    await pyodide.runPythonAsync(`
import os as _os
import warnings as _warnings
_os.environ["MPLBACKEND"] = "Agg"
_warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")
`, { globals });

    for (let index = 0; index < cells.length; index += 1) {
      stdout.length = 0;
      stderr.length = 0;
      postStatus(id, index < targetIndex ? 'running-setup' : 'running');
      const result = await pyodide.runPythonAsync(cells[index], { globals });

      if (index < targetIndex) {
        if (result && typeof result.destroy === 'function') result.destroy();
        await pyodide.runPythonAsync(`
import sys as _sys
if "matplotlib.pyplot" in _sys.modules:
    import matplotlib.pyplot as _plt
    _plt.close("all")
`, { globals });
        continue;
      }

      postStatus(id, 'rendering');
      const images = await captureFigures(pyodide, globals);
      self.postMessage({
        type: 'result',
        id,
        stdout: stdout.join('\n'),
        stderr: stderr.join('\n'),
        result: serialiseResult(result),
        images
      });
    }
  } catch (error) {
    self.postMessage({ type: 'error', id, error: String(error && error.message ? error.message : error) });
  } finally {
    if (globals && typeof globals.destroy === 'function') globals.destroy();
  }
};
