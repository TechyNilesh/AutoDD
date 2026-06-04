"""
Drop-in patch that makes every matplotlib ``savefig('...png')`` ALSO emit a
vector ``...pdf`` next to it - so the paper figures are high quality when zoomed.

It works by being auto-imported at interpreter startup: put this directory on
PYTHONPATH (under the name ``sitecustomize.py``) OR import it explicitly before
running a chart script.

Reproduce the paper's vector figures:

    # from the code/ directory
    mkdir -p /tmp/pdfpatch && cp plotting/savefig_pdf_patch.py /tmp/pdfpatch/sitecustomize.py
    PYTHONPATH=/tmp/pdfpatch python3 plotting/chart_prequential_accuracy.py --base_model all
    PYTHONPATH=/tmp/pdfpatch python3 plotting/chart_cd_diagram.py          --base_model all
    PYTHONPATH=/tmp/pdfpatch python3 plotting/chart_time_memory.py         --base_model all
    PYTHONPATH=/tmp/pdfpatch python3 plotting/chart_detector_selection_combined.py
    PYTHONPATH=/tmp/pdfpatch python3 plotting/chart_window_sensitivity.py
    PYTHONPATH=/tmp/pdfpatch python3 plotting/chart_alpha_sensitivity.py
    PYTHONPATH=/tmp/pdfpatch python3 plotting/chart_lambda_sensitivity.py

The vector PDFs land in plotting/charts/ alongside the PNGs; copy the paper
ones into AutoDD-ICDM-26/images/<subfolder>/.
"""


def _patch():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.figure as mfig
    except Exception as e:  # pragma: no cover
        print('[pdfpatch] matplotlib import failed:', e)
        return
    _orig = mfig.Figure.savefig

    def savefig(self, fname, *a, **k):
        r = _orig(self, fname, *a, **k)
        try:
            s = str(fname)
            if s.lower().endswith('.png'):
                k2 = dict(k)
                k2.pop('format', None)
                _orig(self, s[:-4] + '.pdf', *a, **k2)  # vector copy
        except Exception as e:  # pragma: no cover
            print('[pdfpatch] pdf save failed for', fname, ':', e)
        return r

    mfig.Figure.savefig = savefig
    print('[pdfpatch] active: PNG saves will also emit a vector PDF')


_patch()
