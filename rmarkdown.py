# Build Git Repo

import os
from git import Repo
from config import config
from prefect import task, Flow
import shutil


def knit_rmd_to_html(fp_post):
    """Renders to HTML"""
    cmd = f"Rscript -e 'rmarkdown::render_site(\"{fp_post}\")'"
    os.system(cmd)


@task
def copy_data():
    # Ignore subdirectories
    src_dir = config.get("dir", "output")
    out_dir = os.path.join(config.get("dir", "site"), "data")
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.endswith(".csv")]
    for f in files:
        src_path = os.path.join(src_dir, f)
        out_path = os.path.join(out_dir, f)
        print("Copying from:", src_path, " to:", out_path)
        shutil.copy(src_path, out_path)


@task
def git_commit_push():
    repo = Repo(os.path.expanduser(config.get("dir", "site")))
    repo.git.add(".")
    repo.index.commit("Daily batch run")
    origin = repo.remote(name="origin")
    origin.pull()
    origin.push()


@task
def build_site():
    dir_site = config.get("dir", "site")
    fp_index = os.path.join(dir_site, "index.Rmd")
    knit_rmd_to_html(fp_index)
    knit_rmd_to_html(dir_site)


with Flow("Build Rmd") as flow_rmd:
    c = copy_data()
    b = build_site()
    gcp = git_commit_push(upstream_tasks=[b])

if __name__ == "__main__":
    flow_rmd.run()
