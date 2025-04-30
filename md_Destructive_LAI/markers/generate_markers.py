#!/usr/bin/env python3
"""Generates the svg file of markers used for the leaf scanner
"""

from PIL import Image 
import qrcode
import os
import json
import click


@click.command()
@click.option('--out_dir', "-o", default=".", help="path to output dir. must exist")
@click.option('--json_fp', "-j", default="./markers_list.json", help="json with the names of the cultivar")
def main(out_dir, json_fp):
    svg_template_fp = "./template.svg"
    
    json_data = open(json_fp, "r")
    crops_dict = json.load(json_data)

    def per_plot(crop_name, plot_id, marker_tag, out_svg_fp):
        img = qrcode.make(marker_tag)
        qr_fp = os.path.join(out_dir, f"{marker_tag}.png")
        img.save(qr_fp)
    
        svg_template = open(svg_template_fp, "r")
        svg_data = svg_template.read()
        keyword1 = crop_name.split()[0]
        keyword2 = crop_name.split()[1]  + " " + plot_id # FIXME: this would break if there is no 2 key words
        svg_data = svg_data.replace("Keyword1", keyword1)
        svg_data = svg_data.replace("Keyword2", keyword2)
    
        svg_data = svg_data.replace("LINN_qr_code.png", qr_fp)
    
        out_svg = open(out_svg_fp, "w")
        out_svg.write(svg_data)

    for crop in crops_dict["crops"]:
        crop_name = crop["crop_name"]
        for plot_id in crop["plot_ids"]:
            marker_tag =  f"{crop_name}_{plot_id}"
            out_svg_fp =os.path.join(out_dir, f"{marker_tag}.svg")
            per_plot(crop_name, plot_id, marker_tag, out_svg_fp)

    per_plot("human error", "0", "error", os.path.join(out_dir, "error.svg"))

if __name__ == '__main__':
    main()
