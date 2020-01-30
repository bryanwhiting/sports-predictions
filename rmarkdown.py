# Build Git Repo

import os
from git import Repo
from config import config
from prefect import task, Flow


def knit_rmd_to_html(fp_post):
    """Renders to HTML"""
    cmd = f"Rscript -e 'rmarkdown::render_site(\"{fp_post}\")'"
    os.system(cmd)


@task
def git_commit_push():
    repo = Repo(os.path.expanduser(config.get("dir", "site")))
    repo.git.add(".")
    repo.index.commit("Daily batch run")
    repo.remote(name="origin").push()


@task
def build_site():
    dir_site = config.get("dir", "site")
    fp_index = os.path.join(dir_site, "index.Rmd")
    knit_rmd_to_html(fp_index)


with Flow("Build Rmd") as flow_rmd:
    b = build_site()
    gcp = git_commit_push(upstream_tasks=[b])
