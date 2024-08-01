import os
from livereload import Server, shell

# -------------------------------------------------------------------------
# To use, just execute `python run_live_docs_server.py` in a terminal
# and a local server will run the docs in your browser, automatically
# refreshing/reloading the pages you're working on as they are modified.
# Extremely helpful to see the real output before it gets uploaded, and
# a much smoother experience than constantly running `make html` yourself.
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # establish a local docs server
    svr = Server()

    # get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # change to the script directory
    os.chdir(script_dir)

    # command to rebuild the docs
    refresh_docs = shell("make html", cwd=script_dir)

    # watch for source file changes and trigger rebuild/refresh
    svr.watch("*.rst", refresh_docs, delay=1)
    svr.watch("*.md", refresh_docs, delay=1)
    svr.watch("conf.py", refresh_docs, delay=1)
    svr.watch("api/*", refresh_docs, delay=1)
    svr.watch("user_guide/*", refresh_docs, delay=1)
    svr.watch("developer_guide/*", refresh_docs, delay=1)
    svr.watch("../medmodels/**/*", refresh_docs, delay=1)

    # path from which to serve the docs
    svr.serve(root="_build/html", host="0.0.0.0")
