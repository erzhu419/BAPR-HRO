"""Run unchanged audit scripts against an explicitly pinned local checkout."""
import argparse,importlib.util,os,subprocess,sys
from pathlib import Path

PIN="86ba89c327e5230f1290e8678bebcb589b2465a9"
BUNDLE=Path(__file__).resolve().parent
TASKS={
 "small":("current","probe_small"),
 "residual":("current","diagnose_residual"),
 "board":("board_gap","probe_board"),
 "overlap":("board_gap","probe_overlap"),
 "conditional_overlap":("board_gap","probe_conditional_overlap"),
 "rolling_alight":("board_gap","probe_rolling_alight"),
 "overlap_witnesses":("board_gap","scan_overlap_witnesses"),
 "broad":("algorithm","broad_probe"),
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("task",choices=TASKS)
    args,extra=parser.parse_known_args()
    repo=args.repo.resolve();out=args.out.resolve()
    actual=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if actual!=PIN:raise SystemExit(f"Expected {PIN}; got {actual}. Use a separate checkout at the recorded commit.")
    if repo in out.parents or out==repo:raise SystemExit("Choose an output directory outside the repository.")
    out.mkdir(parents=True,exist_ok=True)
    os.chdir(repo)
    for d in [repo,BUNDLE/"current",BUNDLE/"board_gap",BUNDLE/"algorithm"]:
        sys.path.insert(0,str(d))
    def load(folder,name):
        if name in sys.modules:return sys.modules[name]
        spec=importlib.util.spec_from_file_location(name,BUNDLE/folder/(name+".py"))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module)
        if hasattr(module,"REPO"):module.REPO=repo
        if hasattr(module,"OUT"):module.OUT=out
        return module
    load("current","probe_small")
    load("board_gap","probe_board")
    folder,name=TASKS[args.task]
    module=load(folder,name)
    for value in tuple(sys.modules.values()):
        f=getattr(value,"__file__",None)
        if f and BUNDLE in Path(f).resolve().parents:
            if hasattr(value,"REPO"):value.REPO=repo
            if hasattr(value,"OUT"):value.OUT=out
    sys.argv=[str(BUNDLE/folder/(name+".py")),*extra]
    module.main()

if __name__=="__main__":main()
