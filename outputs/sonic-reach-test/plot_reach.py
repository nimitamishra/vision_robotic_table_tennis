"""Render a static reach trace for browsers where Viser WebGL is unavailable."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / 'samples.jsonl').open()]
    rows = [row for row in rows if row['phase'] in ('reach', 'complete')]
    if len(rows) < 2:
        raise SystemExit('Reach trace has fewer than two samples.')
    report = json.loads((args.run / 'report.json').read_text())
    initial = rows[0]['actual_local_m'][1]
    tmax = max(row['trial_elapsed_s'] for row in rows)
    image = Image.new('RGB', (1320, 850), '#ffffff')
    draw = ImageDraw.Draw(image)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title = ImageFont.truetype(font_path, 25)
    label = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 15)
    draw.text((60, 28), 'Simulation-only right-hand reach', fill='#182438', font=title)
    status = 'PASS' if report.get('short_reach_success', report.get('within_2cm_for_final_second')) else 'NOT PASSED'
    draw.text((60, 70), f'{status}  |  Final hand error: {report.get("mean_final_error_cm", 0):.2f} cm  |  '
                         f'Target-direction progress: {report.get("movement_completion_ratio", 0):.0%}',
              fill='#a3382c' if status != 'PASS' else '#277b50', font=label)
    draw.line((60, 112, 98, 112), fill='#2563bd', width=4)
    draw.text((107, 102), 'Target', fill='#182438', font=small)
    draw.line((213, 112, 251, 112), fill='#e06b26', width=4)
    draw.text((260, 102), 'Measured', fill='#182438', font=small)
    axes = ('Forward x', 'Left/right y', 'Height z')
    for axis in range(3):
        top = 160 + axis * 220
        left, right, bottom = 120, 1235, top + 165
        draw.text((left, top - 27), axes[axis], fill='#182438', font=label)
        values = [100 * (row[key][1][axis] - initial[axis])
                  for row in rows for key in ('actual_local_m', 'target_local_m')]
        bound = max(2.5, max(abs(v) for v in values) * 1.2)

        def point(row, key):
            x = left + (right - left) * row['trial_elapsed_s'] / tmax
            y = top + 82 - 75 * (100 * (row[key][1][axis] - initial[axis])) / bound
            return x, y

        for tick in (-bound, 0, bound):
            y = top + 82 - 75 * tick / bound
            draw.line((left, y, right, y), fill='#d8dee8', width=1)
            draw.text((left - 65, y - 10), f'{tick:+.1f}', fill='#506070', font=small)
        draw.text((left - 73, top + 6), 'cm', fill='#506070', font=small)
        for sec in range(0, int(tmax) + 1, 2):
            x = left + (right - left) * sec / tmax
            draw.line((x, top, x, bottom), fill='#edf0f4', width=1)
            if axis == 2:
                draw.text((x - 8, bottom + 8), str(sec), fill='#506070', font=small)
        for key, color in (('target_local_m', '#2563bd'), ('actual_local_m', '#e06b26')):
            draw.line([point(row, key) for row in rows], fill=color, width=4)
        draw.rectangle((left, top, right, bottom), outline='#8b98aa', width=1)
    draw.text((1150, 823), 'Time (s)', fill='#506070', font=small)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
