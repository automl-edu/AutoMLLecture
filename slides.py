import re
import sys
from pathlib import Path
import subprocess
import multiprocessing


try:
    subprocess.run("pdftk", capture_output=True)
    PDFTK_INSTALLED = True
except FileNotFoundError:
    PDFTK_INSTALLED = False

# This files will be deleted in cleanup()
LATEX_TMP_FILES = [".aux", ".log", ".out", ".synctex.gz", ".nav", ".snm", ".toc"]

GIT_REPO = Path("./")  # Overwritten in main
DST_FOLDER = Path("dst/")  # Overwritten in main

FULL_PDF_NAME = "autoML"
PDF_AUTHOR = "Marius Lindauer"


# Commands
def copy():
    DST_FOLDER.mkdir(parents=True, exist_ok=True)

    for file, week_number, slide_number in iter_all():
        copy_destination = DST_FOLDER / f"w{week_number:02d}_{file.name}"

        print(f"Copy {file} to {copy_destination}", file=sys.stderr)
        with file.open("rb") as d:
            content = d.read()

        with copy_destination.open("wb") as d:
            d.write(content)

#sorts the values(paths) of the dict created in weekly and full slides   
def sort_paths(plist):
    for i in range(1, len(plist)):
        j = i-1
        nxt_element = plist[i]
        #get the slide numbers
        number1 = int(''.join(filter(str.isdigit, nxt_element.name)))
        number2 = int(''.join(filter(str.isdigit, plist[j].name)))
       
        if number1 >= 10:
            number1 /= 10

        if number2 >= 10:
            number2 /= 10
    
        while number2 > number1  and j >= 0:
            plist[j+1] = plist[j]
            j = j - 1
            number2 = int(''.join(filter(str.isdigit, plist[j].name)))
            
            if number2 >= 10:
                number2 /= 10
        plist[j+1] = nxt_element    


def weekly_slides():
    assert_pdftk()
    DST_FOLDER.mkdir(parents=True, exist_ok=True)

    sources = {}
    for file, week_number, slide_number in iter_all():
        if week_number not in sources:
            sources[week_number] = (file.parent.name, [])
        sources[week_number][1].append(file)
    for key in sources:
        sort_paths(sources[key][1])
    
    for week_name, sources in sources.values():
        pdftk(sources, DST_FOLDER / f"{week_name}.pdf", Author=PDF_AUTHOR, Title=week_name.replace("_", " - ").title())


def full_slides():
    assert_pdftk()
    DST_FOLDER.mkdir(parents=True, exist_ok=True)

    sources = {}
    for file, week_number, slide_number in iter_all():
        if week_number not in sources:
            sources[week_number]= (file.parent.name, [])
        sources[week_number][1].append(file)
    for key in sources:
        sort_paths(sources[key][1])
        
    
    key_list= list(sources)
    key_list.sort()
    sources_list= list()
    for i in key_list:
        sources_list.extend(sources[i][1])
    
    pdftk(sources_list, DST_FOLDER / f"{FULL_PDF_NAME}.pdf", Author=PDF_AUTHOR, Title=FULL_PDF_NAME)


def compile_all():
    files = (file for file, _, _ in iter_all(ext="tex"))
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(pdflatex, files)


def compile_git():
    files = (file for file, _, _ in iter_all(ext="tex") if check_git(file.with_suffix(".tex")))
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(pdflatex, files)


def compile_single(week_id, slide_id):
    week_id = int(week_id)
    slide_id = None if slide_id == "*" else int(slide_id)

    def fits_identifier(week, slide):
        if slide_id is None:
            return week == week_id
        return week == week_id and slide == slide_id

    files = (file for file, week, slide in iter_all(ext="tex") if fits_identifier(week, slide))
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(pdflatex, files)


def cleanup():
    for file, week_number, slide_number in iter_all():
        for ext in LATEX_TMP_FILES:
            file.with_suffix(ext).unlink(missing_ok=True)


# Helper functions
def iter_all(ext="pdf"):
    folder_pattern = re.compile("w(\d{2})_")
    slide_pattern = re.compile("t(\d{2,3})_[\w_]+\." + ext)
    for week_folder in GIT_REPO.iterdir():
        week_number = folder_pattern.match(week_folder.name)
        if week_number is None:  # folder does not match mattern
            continue
            
        week_number = int(week_number.group(1))

        for file in week_folder.iterdir():

            slide_number = slide_pattern.match(file.name)
            if slide_number is None:  # Slide does not match pattern (e.g. no pdf, project_exam, ...)
                continue
            slide_number = int(slide_number.group(1))

            yield file.absolute(), week_number, slide_number


def assert_pdftk():
    if not PDFTK_INSTALLED:
        raise ImportError("Requires pdftk for merging slides (https://www.pdflabs.com/tools/pdftk-the-pdf-toolkit/)")


def pdftk(source, out_file, **meta_data) -> int:
    if isinstance(source, (list, tuple, set)):
        args = ["pdftk"] + list(source) + ["cat"]
    else:
        args = ["pdftk", source, "cat"]

    tmp_info = Path("tmp_file.info")
    tmp_pdf = tmp_info.with_suffix(".pdf")
    if meta_data:
        with tmp_info.open("w") as d:
            for key, value in meta_data.items():
                d.write(f"InfoBegin\nInfoKey: {key}\nInfoValue: {value}\n")
        args += ["output", tmp_pdf]
    else:
        args += ["output", out_file]

    print(args, file=sys.stderr)
    proc = subprocess.run(args)

    if proc.returncode != 0:
        print(proc, file=sys.stderr)
    elif meta_data:
        args = ["pdftk", tmp_pdf, "update_info", tmp_info, "output", out_file]
        print(args, file=sys.stderr)
        proc = subprocess.run(args)

        if proc.returncode != 0:
            print(proc, file=sys.stderr)

    tmp_info.unlink(missing_ok=True)
    tmp_pdf.unlink(missing_ok=True)

    return proc.returncode


def pdflatex(source) -> int:
    args = ["pdflatex", "-interaction=nonstopmode", "-output-format=pdf", source]
    print(args, file=sys.stderr)
    proc = subprocess.run(args, cwd=str(source.parent), stdout=subprocess.PIPE)
    if proc.returncode != 0:
        print(proc, file=sys.stderr)
    return proc.returncode


def check_git(file) -> bool:
    proc = subprocess.run(["git", "status", file], stdout=subprocess.PIPE)
    return "nothing to commit" not in proc.stdout.decode()


def main():
    from argparse import ArgumentParser

    # Parameter
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", help="Directory of Repository", default=".", type=str,
                        metavar="<git-repository>")
    parser.add_argument("-d", "--destination", help="Directory of Destination", default="./slides", type=str,
                        metavar="<destination-folder>")

    # Actions
    parser.add_argument("-W", "--weekly", help="Create a pdf for each week", action='store_true')
    parser.add_argument("-A", "--all", help="Create a pdf with all slides", action='store_true')
    parser.add_argument("-C", "--copy", help="Copy all slides into a single folder", action='store_true')
    parser.add_argument("--compile",
                        help="Compile a single file or a single week based on identifiers. Possible values include (1 2, 4 *)",
                        type=str, metavar="<week> <slide>", nargs=2, required=False)
    parser.add_argument("--compile-git", help="Compile slides that have changes according to git",
                        action='store_true')
    parser.add_argument("--compile-all", help="Compile all slides", action='store_true')
    parser.add_argument("--cleanup", help="Cleanup all temporary latex files (aux, log, ...)", action='store_true')

    args = parser.parse_args()

    global GIT_REPO, DST_FOLDER
    GIT_REPO = Path(args.source)
    DST_FOLDER = Path(args.destination)

    _did_smth = False
    if args.compile_all:
        compile_all()
        _did_smth = True
    if args.compile_git and not args.compile_all:
        compile_git()
        _did_smth = True
    if args.compile and not (args.compile_all or args.compile_git):
        compile_single(*args.compile)
        _did_smth = True
    if args.cleanup:
        cleanup()
        _did_smth = True
    if args.weekly:
        weekly_slides()
        _did_smth = True
    if args.all:
        full_slides()
        _did_smth = True
    if args.copy:
        copy()
        _did_smth = True

    if not _did_smth:
        parser.print_usage()


if __name__ == '__main__':
    main()