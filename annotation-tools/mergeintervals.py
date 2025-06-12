import json
import glob
import os

def merge_annotation_files(pattern="annotations_interval*.json",
                          output_file="merged_annotations.json"):
    """
    Merges multiple JSON annotation files (matching 'pattern')
    into a single JSON file named 'output_file'.
    """
    # We'll store the final merged data here.
    # We only keep UrlLocal, UrlYoutube, and halftime from the *first* file we read.
    master_data = {
        "UrlLocal": "",
        "UrlYoutube": "",
        "halftime": "",
        "annotations": []
    }

    # Find all files matching the pattern
    # e.g. annotations_interval1.json, annotations_interval2.json, ...
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching pattern {pattern}")
        return

    first_file = True
    for fname in files:
        print(f"Merging from {fname}...")
        with open(fname, "r") as f:
            data = json.load(f)

            if first_file:
                # Take these fields from the first file we encounter
                master_data["UrlLocal"]   = data.get("UrlLocal", "")
                master_data["UrlYoutube"] = data.get("UrlYoutube", "")
                master_data["halftime"]   = data.get("halftime", "")
                first_file = False

            # Extend annotations
            ann_list = data.get("annotations", [])
            master_data["annotations"].extend(ann_list)

    # Optionally, sort the merged annotations by 'position' (numeric):
    master_data["annotations"].sort(key=lambda x: int(x.get("position", "0")))

    # Write out the merged JSON
    with open(output_file, "w") as out:
        json.dump(master_data, out, indent=4)
    print(f"Done! Merged {len(files)} files into '{output_file}' with {len(master_data['annotations'])} total annotations.")

if __name__ == "__main__":
    # Example usage:
    # 1) Put this script in the same folder as your annotation files.
    # 2) Run: python merge_annotations.py
    #    (which will pick up all "annotations_interval*.json" and produce "merged_annotations.json")
    #
    # If your files have a different naming scheme, adjust the pattern or supply it as an argument.
    merge_annotation_files(
        pattern="annotations_interval*.json",
        output_file="merged_annotations.json"
    )
