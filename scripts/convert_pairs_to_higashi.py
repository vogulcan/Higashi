#!/usr/bin/env python3
"""Convert a 5-column .pairs file into Higashi input files.

Expected input columns:
    chrom1, pos1, chrom2, pos2, cell_label

Supported outputs:
    - higashi_v1:
        data.txt with header:
        cell_id, chrom1, pos1, chrom2, pos2, count
    - higashi_v2:
        filelist.txt plus one per-cell contact file with header:
        chrom1, pos1, chrom2, pos2, count

The converter keeps only intra-chromosomal rows and assigns continuous
numeric cell IDs in order of first appearance. It writes an intermediate
unsorted file and uses external `sort` on cell_id to keep the conversion
scalable for large inputs.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_INPUT = Path("/home/carlos/Clone/schicrna/data/GSM7657694_human_only.pairs")
DEFAULT_V1_OUTPUT = Path(
    "/home/carlos/Clone/schicrna/data/GSM7657694_human_only.higashi_v1.structured.data.txt"
)
DEFAULT_V1_CELL_MAP = Path(
    "/home/carlos/Clone/schicrna/data/GSM7657694_human_only.higashi_v1.cell_map.tsv"
)
DEFAULT_V1_LABEL_INFO = Path(
    "/home/carlos/Clone/schicrna/data/GSM7657694_human_only.higashi_v1.label_info.pickle"
)
DEFAULT_V2_OUTPUT_DIR = Path(
    "/home/carlos/Clone/schicrna/data/GSM7657694_human_only.higashi_v2"
)

V1_HEADER = b"cell_id\tchrom1\tpos1\tchrom2\tpos2\tcount\n"
V2_HEADER = "chrom1\tpos1\tchrom2\tpos2\tcount\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a .pairs file into Higashi input files."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input .pairs file.")
    parser.add_argument(
        "--format",
        choices=("higashi_v1", "higashi_v2"),
        default="higashi_v2",
        help="Higashi input format to write.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "For higashi_v1, output data file. "
            "For higashi_v2, output directory that will contain filelist.txt."
        ),
    )
    parser.add_argument(
        "--cell-map",
        type=Path,
        default=None,
        help="Output cell ID mapping TSV. Defaults next to the chosen output.",
    )
    parser.add_argument(
        "--label-info",
        type=Path,
        default=None,
        help="Output Higashi label_info.pickle. Defaults next to the chosen output.",
    )
    parser.add_argument(
        "--contacts-subdir",
        default="cells",
        help="Relative subdirectory inside the higashi_v2 output directory for per-cell contacts.",
    )
    parser.add_argument(
        "--sort-tempdir",
        type=Path,
        default=None,
        help="Temporary directory for external sort. Defaults to the output directory.",
    )
    parser.add_argument(
        "--sort-buffer-size",
        default=None,
        help="Optional buffer size passed to `sort -S`, for example 4G.",
    )
    parser.add_argument(
        "--sort-parallel",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number passed to `sort --parallel`.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate temporary files.",
    )
    return parser.parse_args()


def ensure_sort_available() -> None:
    if shutil.which("sort") is None:
        raise RuntimeError("Required command `sort` was not found on PATH.")


def sibling_tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def resolve_output_paths(
    args: argparse.Namespace,
) -> tuple[Path | None, Path | None, Path | None, Path, Path, Path]:
    if args.format == "higashi_v1":
        output_path = (args.output or DEFAULT_V1_OUTPUT).resolve()
        if args.cell_map is not None:
            cell_map_path = args.cell_map.resolve()
        elif args.output is not None:
            cell_map_path = output_path.with_name(output_path.stem + ".cell_map.tsv")
        else:
            cell_map_path = DEFAULT_V1_CELL_MAP.resolve()

        if args.label_info is not None:
            label_info_path = args.label_info.resolve()
        elif args.output is not None:
            label_info_path = output_path.with_name(output_path.stem + ".label_info.pickle")
        else:
            label_info_path = DEFAULT_V1_LABEL_INFO.resolve()

        return output_path, None, None, cell_map_path, label_info_path, output_path.parent

    output_dir = (args.output or DEFAULT_V2_OUTPUT_DIR).resolve()
    contacts_subdir = Path(args.contacts_subdir)
    if contacts_subdir.is_absolute():
        raise ValueError("--contacts-subdir must be a relative path.")

    filelist_path = (output_dir / "filelist.txt").resolve()
    contacts_dir = (output_dir / contacts_subdir).resolve()
    cell_map_path = (args.cell_map or output_dir / "cell_map.tsv").resolve()
    label_info_path = (args.label_info or output_dir / "label_info.pickle").resolve()
    return None, filelist_path, contacts_dir, cell_map_path, label_info_path, output_dir


def convert_pairs(
    input_path: Path,
    unsorted_path: Path,
) -> tuple[list[str], int, int, int]:
    cell_to_id: dict[str, int] = {}
    id_to_cell: list[str] = []
    total_rows = 0
    kept_rows = 0
    skipped_inter = 0

    with input_path.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as src, unsorted_path.open(
        "w", encoding="utf-8", buffering=8 * 1024 * 1024
    ) as out:
        for line_number, line in enumerate(src, start=1):
            if not line or line[0] == "#":
                continue

            total_rows += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                raise ValueError(
                    f"Line {line_number} has {len(fields)} columns; expected at least 5."
                )

            chrom1, pos1, chrom2, pos2, cell_label = fields[:5]
            if chrom1 != chrom2:
                skipped_inter += 1
                continue

            cell_id = cell_to_id.get(cell_label)
            if cell_id is None:
                cell_id = len(id_to_cell)
                cell_to_id[cell_label] = cell_id
                id_to_cell.append(cell_label)

            out.write(f"{cell_id}\t{chrom1}\t{pos1}\t{chrom2}\t{pos2}\t1\n")
            kept_rows += 1

    return id_to_cell, total_rows, kept_rows, skipped_inter


def write_cell_map(cell_map_path: Path, id_to_cell: list[str]) -> None:
    tmp_path = sibling_tmp_path(cell_map_path)
    with tmp_path.open("w", encoding="utf-8", buffering=1024 * 1024) as out:
        out.write("cell_id\tcell_label\n")
        for cell_id, cell_label in enumerate(id_to_cell):
            out.write(f"{cell_id}\t{cell_label}\n")
    tmp_path.replace(cell_map_path)


def write_label_info(label_info_path: Path, id_to_cell: list[str]) -> None:
    tmp_path = sibling_tmp_path(label_info_path)
    label_info = {
        "cell_id": list(range(len(id_to_cell))),
        "cell_label": list(id_to_cell),
        "cell_name_higashi": list(id_to_cell),
    }
    with tmp_path.open("wb") as out:
        pickle.dump(label_info, out)
    tmp_path.replace(label_info_path)


def sort_by_cell_id(
    input_path: Path,
    output_path: Path,
    sort_tempdir: Path,
    sort_buffer_size: str | None,
    sort_parallel: int,
    header: bytes | None = None,
) -> None:
    tmp_output = sibling_tmp_path(output_path)
    cmd = ["sort", "-t", "\t", "-k1,1n", f"--parallel={sort_parallel}"]
    if sort_buffer_size:
        cmd.extend(["-S", sort_buffer_size])
    cmd.extend(["-T", str(sort_tempdir), str(input_path)])

    env = os.environ.copy()
    env["LC_ALL"] = "C"

    with tmp_output.open("wb", buffering=8 * 1024 * 1024) as out:
        if header is not None:
            out.write(header)

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        assert proc.stdout is not None
        assert proc.stderr is not None
        while True:
            chunk = proc.stdout.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        return_code = proc.wait()

    if return_code != 0:
        tmp_output.unlink(missing_ok=True)
        raise RuntimeError(f"`sort` failed with exit code {return_code}:\n{stderr}")

    tmp_output.replace(output_path)


def build_contact_paths(contacts_dir: Path, cell_count: int) -> list[Path]:
    return [contacts_dir / f"cell_{cell_id:06d}.contacts.tsv" for cell_id in range(cell_count)]


def write_filelist(filelist_path: Path, contact_paths: list[Path]) -> None:
    tmp_path = sibling_tmp_path(filelist_path)
    with tmp_path.open("w", encoding="utf-8", buffering=1024 * 1024) as out:
        for contact_path in contact_paths:
            out.write(f"{contact_path.resolve()}\n")
    tmp_path.replace(filelist_path)


def split_sorted_contacts_into_v2(
    sorted_path: Path,
    contacts_dir: Path,
    cell_count: int,
) -> list[Path]:
    contact_paths = build_contact_paths(contacts_dir, cell_count)
    current_cell_id: int | None = None
    current_out = None
    current_tmp_path: Path | None = None

    try:
        with sorted_path.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as src:
            for line_number, line in enumerate(src, start=1):
                if not line:
                    continue

                fields = line.rstrip("\n").split("\t")
                if len(fields) != 6:
                    raise ValueError(
                        f"Sorted temporary line {line_number} has {len(fields)} columns; expected 6."
                    )

                cell_id = int(fields[0])
                if not 0 <= cell_id < cell_count:
                    raise ValueError(
                        f"Sorted temporary line {line_number} has out-of-range cell_id {cell_id}."
                    )

                if cell_id != current_cell_id:
                    if current_out is not None and current_cell_id is not None and current_tmp_path is not None:
                        current_out.close()
                        current_tmp_path.replace(contact_paths[current_cell_id])

                    current_cell_id = cell_id
                    current_tmp_path = sibling_tmp_path(contact_paths[cell_id])
                    current_out = current_tmp_path.open(
                        "w",
                        encoding="utf-8",
                        buffering=1024 * 1024,
                    )
                    current_out.write(V2_HEADER)

                assert current_out is not None
                current_out.write("\t".join(fields[1:]) + "\n")

        if current_out is not None and current_cell_id is not None and current_tmp_path is not None:
            current_out.close()
            current_tmp_path.replace(contact_paths[current_cell_id])
    finally:
        if current_out is not None and not current_out.closed:
            current_out.close()

    return contact_paths


def main() -> int:
    args = parse_args()
    ensure_sort_available()

    input_path = args.input.resolve()
    output_path, filelist_path, contacts_dir, cell_map_path, label_info_path, default_sort_root = (
        resolve_output_paths(args)
    )
    sort_tempdir = (args.sort_tempdir or default_sort_root).resolve()

    cell_map_path.parent.mkdir(parents=True, exist_ok=True)
    label_info_path.parent.mkdir(parents=True, exist_ok=True)
    sort_tempdir.mkdir(parents=True, exist_ok=True)

    if args.format == "higashi_v1":
        assert output_path is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        unsorted_path = output_path.with_name(output_path.name + ".unsorted.tmp")
        sorted_path = None
    else:
        assert filelist_path is not None
        assert contacts_dir is not None
        filelist_path.parent.mkdir(parents=True, exist_ok=True)
        contacts_dir.mkdir(parents=True, exist_ok=True)
        unsorted_path = filelist_path.with_name("contacts.by_cell.unsorted.tmp")
        sorted_path = filelist_path.with_name("contacts.by_cell.sorted.tmp")

    unsorted_path.unlink(missing_ok=True)
    if sorted_path is not None:
        sorted_path.unlink(missing_ok=True)

    try:
        id_to_cell, total_rows, kept_rows, skipped_inter = convert_pairs(input_path, unsorted_path)
        write_cell_map(cell_map_path, id_to_cell)
        write_label_info(label_info_path, id_to_cell)

        if args.format == "higashi_v1":
            assert output_path is not None
            sort_by_cell_id(
                input_path=unsorted_path,
                output_path=output_path,
                sort_tempdir=sort_tempdir,
                sort_buffer_size=args.sort_buffer_size,
                sort_parallel=args.sort_parallel,
                header=V1_HEADER,
            )
        else:
            assert filelist_path is not None
            assert contacts_dir is not None
            assert sorted_path is not None
            sort_by_cell_id(
                input_path=unsorted_path,
                output_path=sorted_path,
                sort_tempdir=sort_tempdir,
                sort_buffer_size=args.sort_buffer_size,
                sort_parallel=args.sort_parallel,
                header=None,
            )
            contact_paths = split_sorted_contacts_into_v2(
                sorted_path=sorted_path,
                contacts_dir=contacts_dir,
                cell_count=len(id_to_cell),
            )
            write_filelist(filelist_path, contact_paths)
    finally:
        if not args.keep_temp:
            unsorted_path.unlink(missing_ok=True)
            if sorted_path is not None:
                sorted_path.unlink(missing_ok=True)

    print(f"input: {input_path}", file=sys.stderr)
    print(f"format: {args.format}", file=sys.stderr)
    if output_path is not None:
        print(f"output: {output_path}", file=sys.stderr)
    if filelist_path is not None:
        print(f"filelist: {filelist_path}", file=sys.stderr)
        print(f"contacts_dir: {contacts_dir}", file=sys.stderr)
    print(f"cell_map: {cell_map_path}", file=sys.stderr)
    print(f"label_info: {label_info_path}", file=sys.stderr)
    print(f"total_rows: {total_rows}", file=sys.stderr)
    print(f"kept_intra_rows: {kept_rows}", file=sys.stderr)
    print(f"skipped_inter_rows: {skipped_inter}", file=sys.stderr)
    print(f"num_cells: {len(id_to_cell)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
