import os
import random

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler

from PIL import Image, ImageDraw, ImageFont

from oliark_io import file_read
from oliark_io import json_read
from oliark_img import img_resize
from oliark_llm import llm_reply


## vars init
body_text_size = 30
subtitle_text_size = 36

vault = '/home/ubuntu/vault'
llms_path = f'{vault}/llms'
fonts_folderpath = f'{vault}/fonts'
font_helvetica_regular_filepath = f'{fonts_folderpath}/helvetica/Helvetica.ttf' 
font_helvetica_bold_filepath = f'{fonts_folderpath}/helvetica/Helvetica-Bold.ttf' 

model = f'{llms_path}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf' 
model_validator_filepath = f'{llms_path}/Llama-3-Partonus-Lynx-8B-Intstruct-Q4_K_M.gguf'

proj_name = 'terrawhisper'
db_path = f'{vault}/{proj_name}/database/{proj_name}'
encyclopedia_folderpath = f'{vault}/{proj_name}/encyclopedia'
jsons_backup_folderpath = f'{vault}/{proj_name}/encyclopedia/jsons-backup'
jsons_folderpath = f'{vault}/{proj_name}/encyclopedia/jsons'
jsons_folderpath = jsons_folderpath
tmp_data_folderpath = f'{encyclopedia_folderpath}/temp-data'
pdf_folderpath = f'{encyclopedia_folderpath}/pdf'

a4_w = 2480
a4_h = 3508

cell_size = 80
grid_col_num = int(a4_w / cell_size)
grid_row_num = int(a4_h / cell_size) + 1

col_num = 3
col_px = 3 * cell_size
col_width = (a4_w - col_px*2) / col_num

margin_mul = 0.75

mo = int(300*margin_mul)
mi = int(500*margin_mul)
mt = int(500*margin_mul)
mb = int(800*margin_mul)

p1_ml = mo
p1_mt = mt
p1_mr = mi
p1_mb = mb

p2_ml = mi
p2_mt = mt
p2_mr = mo
p2_mb = mb


def a4_draw_grid(draw):
    for i in range(grid_col_num+1):
        x_1 = cell_size*i
        y_1 = 0
        x_2 = cell_size*i
        y_2 = a4_h
        draw.line((x_1, y_1, x_2, y_2), fill='#ff00ff', width=4)
    for i in range(grid_row_num+1):
        x_1 = 0
        y_1 = cell_size*i
        x_2 = a4_w
        y_2 = cell_size*i
        draw.line((x_1, y_1, x_2, y_2), fill='#ff00ff', width=4)

def a4_draw_guides(draw):
    for i in range(col_num+1):
        x_1 = int(col_width*i) + col_px
        y_1 = 0
        x_2 = int(col_width*i) + col_px
        y_2 = a4_h
        draw.line((x_1, y_1, x_2, y_2), fill='#00ffff', width=8)

def draw_ailment_title(draw, line, x_curr, y_curr):
    font_size = 160
    font = ImageFont.truetype(font_helvetica_bold_filepath, font_size)
    draw.text((x_curr, y_curr), line, '#0000', font=font)
    y_curr = y_start
    return y_curr

def draw_subtitle(draw, line, x_curr, y_curr, font_size):
    font = ImageFont.truetype(font_helvetica_bold_filepath, font_size)
    draw.text((x_curr, y_curr), line, '#000000', font=font)
    y_curr += font_size
    return y_curr

y_start = 7*cell_size
x_index = 0

def draw_page_ailment(ailment_filepath, page_i, regen=False):
    ailment_slug = json_filepath.split('/')[-1].split('.')[0]
    ailment_name = ailment_slug.replace('-', ' ')
    print(ailment_name)

    image_filepath_out = f'output/{page_i}-{ailment_slug}.jpg'

    if not regen:
        if os.path.exists(image_filepath_out): 
            return

    data = json_read(ailment_filepath)

    img = Image.new('RGB', (a4_w, a4_h), color='white') 
    draw = ImageDraw.Draw(img)

    x_curr = p1_ml
    y_curr = p1_mt

    inner_width = a4_w - p1_ml - p1_mr
    inner_col_num = 3
    inner_col_width = inner_width / inner_col_num
    inner_col_gap = cell_size

    margin_left = p1_ml
    margin_top = p1_mt
    margin_right = p1_mr
    margin_bottom = p1_mb

    if 0:
        a4_draw_grid(draw)
        ## margin guides
        draw.line((margin_left, 0, margin_left, a4_h), fill='#00ffff', width=8)
        draw.line((0, margin_top, a4_w, margin_top), fill='#00ffff', width=8)
        draw.line((a4_w - margin_right, 0, a4_w - margin_right, a4_h), fill='#00ffff', width=8)
        draw.line((0, a4_h - margin_bottom, a4_w, a4_h - margin_bottom), fill='#00ffff', width=8)
        for i in range(inner_col_num+1):
            x1 = margin_left + inner_col_width*i + inner_col_gap//2
            draw.line((x1, margin_top, x1, a4_h - margin_bottom), fill='#00ffff', width=8)
            x1 = margin_left + inner_col_width*i - inner_col_gap//2
            draw.line((x1, margin_top, x1, a4_h - margin_bottom), fill='#00ffff', width=8)
            x1 = margin_left + inner_col_width*i
            draw.line((x1, margin_top, x1, a4_h - margin_bottom), fill='#ffff00', width=8)

    ## title
    y_start = y_curr
    x_curr = margin_left + inner_col_width*0 + inner_col_gap//2
    y_curr = draw_ailment_title(draw, ailment_name.title(), x_curr, y_curr)
    y_curr += cell_size//2
    line = data['system']
    font_size = 36
    font = ImageFont.truetype(font_helvetica_bold_filepath, font_size)
    _, _, line_w, _ = font.getbbox(line)
    x = int(a4_w - margin_right - line_w - inner_col_gap//2)
    y = int(y_start + cell_size*0.75)
    draw.text((x, y), line, '#000000', font=font)
    line = f"({data['organ']})"
    font_size = 36
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    _, _, line_w, _ = font.getbbox(line)
    x = int(a4_w - margin_right - line_w - inner_col_gap//2)
    y += int(cell_size*0.5)
    draw.text((x, y), line, '#000000', font=font)
    x1 = margin_left + inner_col_gap//2
    x2 = a4_w - margin_right - inner_col_gap//2
    draw.line((x1, y_curr - 8, x2, y_curr - 8), fill='#000000', width=16)
    y_curr += cell_size//2
    x1 = margin_left + inner_col_width*2
    draw.line((x1, y_curr, x1, a4_h - margin_bottom), fill='#000000', width=1)
    y_start_2 = y_curr

    ## definition
    text = data['definition']
    x_curr = margin_left + inner_col_width*0 + inner_col_gap//2
    text_w = inner_col_width*2 - inner_col_gap
    font_size = 42

    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    lines = []
    line_curr = ''
    for word in text.split(' '):
        _, _, word_w, word_h = font.getbbox(word)
        _, _, line_curr_w, line_curr_h = font.getbbox(line_curr)
        if line_curr_w + word_w < text_w:
            line_curr += f'{word} '
        else:
            lines.append(line_curr.strip())
            line_curr = f'{word} '
    lines.append(line_curr)
    line_height = 1.2
    i = 0
    for line in lines:
        y_line = y_curr + font_size*line_height*i
        draw.text((x_curr, y_line), line, '#000000', font=font)
        i += 1
    y_curr = y_line + font_size
    y_curr += cell_size//2

    ## image
    # TODO

    plant_name_scientific = data['plants'][0]['plant_name_scientific']
    plant_slug = plant_name_scientific.strip().lower().replace(' ', '-')
    plant_image_filepath_out = f'images/{page_i}-{ailment_slug}-{plant_slug}.jpg'
    img_x = int(x_curr)
    img_y = int(y_curr)
    img_w = int(inner_col_width*2 - inner_col_gap)
    img_h = int(cell_size*8)
    foreground = Image.open(plant_image_filepath_out)
    foreground = img_resize(foreground, img_w, img_h)
    img.paste(foreground, (img_x, img_y))
    y_curr += img_h
    y_curr += cell_size
    y_start = y_curr
    min_lines = 7
    max_lines = 10

    ## separator vertical
    x1 = margin_left + inner_col_width*1
    draw.line((x1, y_curr, x1, a4_h - margin_bottom), fill='#000000', width=1)

    ## causes 
    font_size = 48
    y_curr = draw_subtitle(draw, 'Causes', x_curr, y_curr, font_size)
    y_curr += cell_size//2
    ## causes list
    font_size = body_text_size
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    line_height = 1.5
    ellipse_size = int(font_size*0.33)
    i = 0
    plants = data['causes']
    for obj in plants[:10]:
        plant_name_scientific = obj['name']
        plant_mentions = str(obj['mentions'])
        x1 = x_curr
        y1 = y_curr + font_size*line_height*i
        x2 = x1 + ellipse_size
        y2 = y1 + ellipse_size
        draw.ellipse((x1, y1 + int(ellipse_size*1.33), x2, y2 + int(ellipse_size*1.33)), fill='black')
        draw.text((x1 + font_size, y1), plant_name_scientific, '#000000', font=font)
        _, _, plant_mentions_w, _ = font.getbbox(plant_mentions)
        draw.text((x1 + inner_col_width - cell_size - plant_mentions_w, y1), plant_mentions, '#000000', font=font)
        i += 1
    y_curr = y1 + font_size
    y_curr += cell_size//2

    ## symptoms 
    font_size = 48
    y_curr = draw_subtitle(draw, 'Symptoms', x_curr, y_curr, font_size)
    y_curr += cell_size//2
    ## symptoms list
    font_size = body_text_size
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    line_height = 1.5
    ellipse_size = int(font_size*0.33)
    i = 0
    plants = data['symptoms']
    for obj in plants[:10]:
        plant_name_scientific = obj['name']
        plant_mentions = str(obj['mentions'])
        x1 = x_curr
        y1 = y_curr + font_size*line_height*i
        x2 = x1 + ellipse_size
        y2 = y1 + ellipse_size
        draw.ellipse((x1, y1 + int(ellipse_size*1.33), x2, y2 + int(ellipse_size*1.33)), fill='black')
        draw.text((x1 + font_size, y1), plant_name_scientific, '#000000', font=font)
        _, _, plant_mentions_w, _ = font.getbbox(plant_mentions)
        draw.text((x1 + inner_col_width - cell_size - plant_mentions_w, y1), plant_mentions, '#000000', font=font)
        i += 1
    y_curr = y1 + font_size
    y_curr += cell_size//2

    # prevention
    text = data['preventions']
    x_curr = margin_left + inner_col_width*1 + inner_col_gap//2
    y_curr = y_start
    font_size = 48
    y_curr = draw_subtitle(draw, 'Preventions', x_curr, y_curr, font_size)
    y_curr += cell_size//2
    text_w = inner_col_width*1 - inner_col_gap
    font_size = 30
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    lines = []
    line_curr = ''
    for word in text.split(' '):
        _, _, word_w, word_h = font.getbbox(word)
        _, _, line_curr_w, line_curr_h = font.getbbox(line_curr)
        if line_curr_w + word_w < text_w:
            line_curr += f'{word} '
        else:
            lines.append(line_curr.strip())
            line_curr = f'{word} '
    lines.append(line_curr)
    line_height = 1.2
    i = 0
    for line in lines:
        y_line = y_curr + font_size*line_height*i
        draw.text((x_curr, y_line), line, '#000000', font=font)
        i += 1
        print(line)
    y_curr = y_line + font_size
    y_curr += cell_size

    # complications
    text = data['complications']
    font_size = 48
    y_curr = draw_subtitle(draw, 'Complication', x_curr, y_curr, font_size)
    y_curr += cell_size//2
    text_w = inner_col_width*1 - inner_col_gap
    font_size = 30
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    lines = []
    line_curr = ''
    for word in text.split(' '):
        _, _, word_w, word_h = font.getbbox(word)
        _, _, line_curr_w, line_curr_h = font.getbbox(line_curr)
        if line_curr_w + word_w < text_w:
            line_curr += f'{word} '
        else:
            lines.append(line_curr.strip())
            line_curr = f'{word} '
    lines.append(line_curr)
    line_height = 1.2
    i = 0
    for line in lines:
        y_line = y_curr + font_size*line_height*i
        draw.text((x_curr, y_line), line, '#000000', font=font)
        i += 1
        print(line)
    y_curr = y_line + font_size
    y_curr += cell_size

    ## herbs
    x_curr = margin_left + inner_col_width*2 + inner_col_gap//2
    y_curr = y_start_2
    ## herbs title
    font_size = 48
    y_curr = draw_subtitle(draw, 'Herbs', x_curr, y_curr, font_size)
    y_curr += cell_size//2
    ## herbs list
    font_size = body_text_size
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    line_height = 1.5
    ellipse_size = int(font_size*0.33)
    i = 0
    plants = data['plants']
    for obj in plants[:10]:
        plant_name_scientific = obj['plant_name_scientific']
        plant_mentions = str(obj['plant_mentions'])
        x1 = x_curr
        y1 = y_curr + font_size*line_height*i
        x2 = x1 + ellipse_size
        y2 = y1 + ellipse_size
        draw.ellipse((x1, y1 + int(ellipse_size*1.33), x2, y2 + int(ellipse_size*1.33)), fill='black')
        draw.text((x1 + font_size, y1), plant_name_scientific, '#000000', font=font)
        _, _, plant_mentions_w, _ = font.getbbox(plant_mentions)
        draw.text((x1 + inner_col_width - cell_size - plant_mentions_w, y1), plant_mentions, '#000000', font=font)
        i += 1
    y_curr = y1 + font_size
    y_curr += cell_size//2
    ## herbs desc
    text = data['plants_desc']
    text_w = inner_col_width*1 - inner_col_gap
    font_size = 30
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    lines = []
    line_curr = ''
    for word in text.split(' '):
        _, _, word_w, word_h = font.getbbox(word)
        _, _, line_curr_w, line_curr_h = font.getbbox(line_curr)
        if line_curr_w + word_w < text_w:
            line_curr += f'{word} '
        else:
            lines.append(line_curr.strip())
            line_curr = f'{word} '
    lines.append(line_curr)
    line_height = 1.2
    i = 0
    for line in lines:
        y_line = y_curr + font_size*line_height*i
        draw.text((x_curr, y_line), line, '#000000', font=font)
        i += 1
        print(line)
    y_curr = y_line + font_size
    y_curr += cell_size

    # when to seek medical attention
    pass

    ## page number left
    x_curr = margin_left + inner_col_width*0 + inner_col_gap//2
    font_size = 32
    line = str(page_i)
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    _, _, _, line_h = font.getbbox(line)
    draw.text((x_curr, a4_h - margin_bottom + line_h + cell_size), line, '#000000', font=font)

    # img.show()
    # img.save(f'output/p1.jpg')
    img.save(image_filepath_out)


def draw_herbs_paragraph(draw, text, x_curr, y_curr):
    font_size = body_text_size
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    lines = []
    line_curr = ''
    for word in text.split(' '):
        _, _, word_w, word_h = font.getbbox(word)
        _, _, line_curr_w, line_curr_h = font.getbbox(line_curr)
        if line_curr_w + word_w < col_width - cell_size:
            line_curr += f'{word} '
        else:
            lines.append(line_curr.strip())
            line_curr = f'{word} '
    lines.append(line_curr)
    line_height = 1.2
    i = 0
    for line in lines:
        y_line = y_curr + font_size*line_height*i
        draw.text((x_curr, y_line), line, '#000000', font=font)
        i += 1
    y_curr = y_line + font_size
    y_curr += cell_size//2
    return y_curr

def draw_page_herbs(ailment_filepath, page_i, regen=False):
    ailment_slug = json_filepath.split('/')[-1].split('.')[0]
    ailment_name = ailment_slug.replace('-', ' ')
    print(ailment_name)

    image_filepath_out = f'output/{page_i}-{ailment_slug}.jpg'

    if not regen:
        if os.path.exists(image_filepath_out): 
            return

    data = json_read(ailment_filepath)

    img = Image.new('RGB', (a4_w, a4_h), color='white')
    draw = ImageDraw.Draw(img)

    text_area_x1 = p2_ml
    text_area_y1 = p2_mt
    text_area_x2 = a4_w - p2_mr
    text_area_y2 = a4_h - p2_mb
    text_area_w = text_area_x2 - text_area_x1
    text_area_h = text_area_y2 - text_area_y1

    col_num = 3
    col_w = text_area_w / col_num

    row_num = 4
    row_h = text_area_h / row_num

    line_num = 15
    line_h = row_h / line_num

    col_gap = line_h
    row_gap = line_h

    body_font_size = line_h

    if 0:
        for i in range(row_num * line_num + 1):
            x1 = text_area_x1
            y1 = text_area_y1 + line_h*i*line_spacing
            x2 = text_area_x2
            y2 = text_area_y1 + line_h*i*line_spacing
            draw.line((x1, y1, x2, y2), fill='#00ff00', width=4)
        for i in range(col_num + 1):
            x1 = text_area_x1 + col_w*i
            y1 = text_area_y1
            x2 = text_area_x1 + col_w*i
            y2 = text_area_y2
            draw.line((x1, y1, x2, y2), fill='#00ffff', width=4)
            draw.line((x1-col_gap, y1, x2-col_gap, y2), fill='#00ffff', width=4)
        for i in range(row_num + 1):
            x1 = text_area_x1
            y1 = text_area_y1 + row_h*i*lien_spacing
            x2 = text_area_x2
            y2 = text_area_y1 + row_h*i*lien_spacing
            draw.line((x1, y1, x2, y2), fill='#00ffff', width=4)
            draw.line((x1, y1-row_gap, x2, y2-row_gap), fill='#00ffff', width=4)
        draw.line((text_area_x1, text_area_y1, text_area_x1, text_area_y2), fill='#ff0000', width=4)
        draw.line((text_area_x2, text_area_y1, text_area_x2, text_area_y2), fill='#ff0000', width=4)
        draw.line((text_area_x1, text_area_y1, text_area_x2, text_area_y1), fill='#ff0000', width=4)
        draw.line((text_area_x1, text_area_y2, text_area_x2, text_area_y2), fill='#ff0000', width=4)
            

    # draw_separators_vertical(draw)
    '''
    x_1 = int(col_width*1) + col_px
    y_1 = y_start
    x_2 = int(col_width*1) + col_px
    y_2 = 3040
    draw.line((x_1, y_1, x_2, y_2), fill='#cdcdcd', width=2)
    x_1 = int(col_width*2) + col_px
    y_1 = y_start
    x_2 = int(col_width*2) + col_px
    y_2 = 3040
    draw.line((x_1, y_1, x_2, y_2), fill='#cdcdcd', width=2)
    '''

    ## draw images
    img_size = int(col_w - col_gap)
    for i in range(3):
        plant_name_scientific = data['plants'][i]['plant_name_scientific']
        plant_slug = plant_name_scientific.lower().strip().replace(' ', '-')
        plant_image_filepath_out = f'images/{page_i}-{ailment_slug}-plant-{plant_slug}.jpg'
        col_index = 0
        x1 = int(text_area_x1 + col_w*i)
        y1 = int(text_area_y1)
        foreground = Image.open(plant_image_filepath_out)
        foreground = img_resize(foreground, img_size, img_size)
        img.paste(foreground, (x1, y1))

    ## draw text
    x_index = 0
    remedies = data['remedies']
    remedy_num = 1
    remedy_num = 3
    for remedy_i, remedy in enumerate(remedies[:remedy_num]):
        x_curr = text_area_x1 + col_w*remedy_i
        y_curr = text_area_y1 + row_h
        print('###########################################')
        print(x_index)
        print(x_curr)
        print('###########################################')

        ## plant title
        remedy_plant_name_scientific = remedy['plant_name_scientific']
        line = remedy_plant_name_scientific.title()
        font_size = body_font_size
        font = ImageFont.truetype(font_helvetica_bold_filepath, font_size)
        draw.text((x_curr, y_curr), line, '#000000', font=font)
        y_curr += font_size + cell_size//2
        ## use paragraph
        remedy_intro = remedy['attributes']['intro']
        y_curr = draw_herbs_paragraph(draw, remedy_intro, x_curr, y_curr)

        ## constituents subtitle
        line = 'Constituents'.title()
        font_size = body_font_size
        font = ImageFont.truetype(font_helvetica_bold_filepath, font_size)
        draw.text((x_curr, y_curr), line, '#000000', font=font)
        y_curr += font_size + line_h
        ## constituents list
        lines = [remedy['name'].capitalize() for remedy in remedy['attributes']['constituents']][:random.randint(3, 5)]
        font_size = body_text_size
        font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
        line_height = 1.5
        ellipse_size = font_size//2
        i = 0
        for line in lines:
            print(line)
            x1 = x_curr
            y1 = y_curr + font_size*line_height*i
            x2 = x1 + ellipse_size
            y2 = y1 + ellipse_size
            draw.ellipse((x1, y1 + int(font_size*0.25), x2, y2 + int(font_size*0.25)), fill='black')
            draw.text((x1 + 32, y1), line, '#000000', font=font)
            i += 1
        y_curr = y1 + font_size
        y_curr += cell_size//2

        ## parts subtitle
        line = 'Parts'.title()
        font_size = subtitle_text_size
        font = ImageFont.truetype(font_helvetica_bold_filepath, font_size)
        draw.text((x_curr, y_curr), line, '#000000', font=font)
        y_curr += font_size + cell_size//2
        ## parts list
        lines = [remedy['name'].capitalize() for remedy in remedy['attributes']['parts']][:random.randint(3, 5)]
        font_size = body_text_size
        font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
        line_height = 1.5
        ellipse_size = font_size//2
        i = 0
        for line in lines:
            x1 = x_curr
            y1 = y_curr + font_size*line_height*i
            x2 = x1 + ellipse_size
            y2 = y1 + ellipse_size
            draw.ellipse((x1, y1 + int(font_size*0.25), x2, y2 + int(font_size*0.25)), fill='black')
            draw.text((x1 + 32, y1), line, '#000000', font=font)
            i += 1
        y_curr = y1 + font_size
        y_curr += cell_size//2

        ## preparations subtitle
        line = 'Preparations'.title()
        font_size = subtitle_text_size
        font = ImageFont.truetype(font_helvetica_bold_filepath, font_size)
        draw.text((x_curr, y_curr), line, '#000000', font=font)
        y_curr += font_size + cell_size//2
        ## preparations list
        lines = [remedy['name'].capitalize() for remedy in remedy['attributes']['preparations']][:random.randint(3, 5)]
        font_size = body_text_size
        font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
        line_height = 1.5
        ellipse_size = font_size//2
        i = 0
        for line in lines:
            x1 = x_curr
            y1 = y_curr + font_size*line_height*i
            x2 = x1 + ellipse_size
            y2 = y1 + ellipse_size
            draw.ellipse((x1, y1 + int(font_size*0.25), x2, y2 + int(font_size*0.25)), fill='black')
            draw.text((x1 + 32, y1), line, '#000000', font=font)
            i += 1
        y_curr = y1 + font_size
        y_curr += cell_size//2

        ## precautions subtitle
        line = 'Precautions'.title()
        font_size = subtitle_text_size
        font = ImageFont.truetype(font_helvetica_bold_filepath, font_size)
        draw.text((x_curr, y_curr), line, '#000000', font=font)
        y_curr += font_size + cell_size//2
        ## precautions paragraph
        remedy_precautions = remedy['attributes']['precautions']
        y_curr = draw_herbs_paragraph(draw, remedy_precautions, x_curr, y_curr)

        x_index += 1 

    ## page number left
    font_size = 32
    line = str(page_i)
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    _, _, line_w, _ = font.getbbox(line)
    x1 = a4_w - 3*cell_size - line_w - cell_size//2
    draw.text((x1, 40*cell_size), line, '#000000', font=font)

    img.save(image_filepath_out)
    # img.show()

def preview_full(json_filepath, page_i):
    ailment_slug = json_filepath.split('/')[-1].split('.')[0]
    ailment_name = ailment_slug.strip().replace('-', ' ')
    separator_width = 8
    img = Image.new('RGB', (a4_w*2, a4_h), color='white')
    draw = ImageDraw.Draw(img)
    img_1 = Image.open(f'output/{page_i}-{ailment_name}.jpg')
    img_2 = Image.open(f'output/{page_i+1}-{ailment_name}.jpg')
    img.paste(img_1, (0, 0))
    img.paste(img_2, (a4_w, 0))
    draw.line((a4_w - separator_width//2, 0, a4_w - separator_width//2, a4_h), fill='#000000', width=separator_width)

    img.save(f'output/full.jpg')
    img.show()

def draw_ailment_image(json_filepath, page_i):
    ailment_slug = json_filepath.split('/')[-1].split('.')[0]
    ailment_name = ailment_slug.replace('-', ' ')
    print(ailment_name)

    data = json_read(json_filepath)
    plant_name_scientific = data['plants'][0]['plant_name_scientific']
    plant_slug = plant_name_scientific.lower().strip().replace(' ', '-')
    plant_image_filepath_out = f'images/{page_i}-{ailment_slug}-{plant_slug}.jpg'

    if not os.path.exists(plant_image_filepath_out): 
        prompt = f'''
            {plant_name_scientific},
            outdoor,
            natural light,
            depth of field, bokeh, 
            high resolution, cinematic
        '''
        image = pipe(prompt=prompt, num_inference_steps=25).images[0]
        image.save(plant_image_filepath_out)

def draw_ailment_plants_images(json_filepath, page_i):
    ailment_slug = json_filepath.split('/')[-1].split('.')[0]
    ailment_name = ailment_slug.replace('-', ' ')
    print(ailment_name)

    data = json_read(json_filepath)

    ## gen images
    for i in range(3):
        plant_name_scientific = data['plants'][i]['plant_name_scientific']
        plant_slug = plant_name_scientific.lower().strip().replace(' ', '-')
        plant_image_filepath_out = f'images/{page_i}-{ailment_slug}-plant-{plant_slug}.jpg'
        if not os.path.exists(plant_image_filepath_out):
            prompt = f'''
                {plant_name_scientific},
                outdoor,
                natural light,
                depth of field, bokeh, 
                high resolution, cinematic
            '''
            image = pipe(prompt=prompt, num_inference_steps=25).images[0]
            image.save(plant_image_filepath_out)

def draw_page(ailment_filepath, page_i, side, regen=False):
    ailment_slug = json_filepath.split('/')[-1].split('.')[0]
    ailment_name = ailment_slug.replace('-', ' ')
    print(ailment_name)

    image_filepath_out = f'output/{page_i}-{ailment_slug}.jpg'

    if not regen:
        if os.path.exists(image_filepath_out): 
            return

    data = json_read(ailment_filepath)

    img = Image.new('RGB', (a4_w, a4_h), color='white')
    draw = ImageDraw.Draw(img)

    p_mt = mt
    p_mb = mb
    if side == 'l':
        p_ml = mi
        p_mr = mo
    else:
        p_ml = mo
        p_mr = mi

    text_area_x1 = p_ml
    text_area_y1 = p_mt
    text_area_x2 = a4_w - p_mr
    text_area_y2 = a4_h - p_mb
    text_area_w = text_area_x2 - text_area_x1
    text_area_h = text_area_y2 - text_area_y1

    col_num = 2
    col_w = text_area_w / col_num

    row_num = 4
    row_h = text_area_h / row_num

    line_num = 13
    line_spacing = 1.3
    line_h = row_h / line_num

    body_font_size = line_h / line_spacing
    body_line_spacing = line_spacing

    col_gap = line_h
    row_gap = line_h

    if 0:
        for i in range(row_num * line_num + 1):
            x1 = text_area_x1
            y1 = text_area_y1 + line_h*i
            x2 = text_area_x2
            y2 = text_area_y1 + line_h*i
            draw.line((x1, y1, x2, y2), fill='#00ff00', width=4)
        for i in range(col_num + 1):
            x1 = text_area_x1 + col_w*i
            y1 = text_area_y1
            x2 = text_area_x1 + col_w*i
            y2 = text_area_y2
            draw.line((x1, y1, x2, y2), fill='#00ffff', width=4)
            draw.line((x1-col_gap, y1, x2-col_gap, y2), fill='#00ffff', width=4)
        for i in range(row_num + 1):
            x1 = text_area_x1
            y1 = text_area_y1 + row_h*i
            x2 = text_area_x2
            y2 = text_area_y1 + row_h*i
            draw.line((x1, y1, x2, y2), fill='#00ffff', width=4)
            draw.line((x1, y1 - row_gap, x2, y2 - row_gap), fill='#00ffff', width=4)
        draw.line((text_area_x1, text_area_y1, text_area_x1, text_area_y2), fill='#ff0000', width=4)
        draw.line((text_area_x2, text_area_y1, text_area_x2, text_area_y2), fill='#ff0000', width=4)
        draw.line((text_area_x1, text_area_y1, text_area_x2, text_area_y1), fill='#ff0000', width=4)
        draw.line((text_area_x1, text_area_y2, text_area_x2, text_area_y2), fill='#ff0000', width=4)
            
    ## title
    text = ailment_name.title()
    font_size_mul = 1.8
    font_size = body_font_size*font_size_mul
    font = ImageFont.truetype(font_helvetica_bold_filepath, font_size)
    lines = []
    line_curr = ''
    for word in text.split(' '):
        _, _, word_w, word_h = font.getbbox(word)
        _, _, line_curr_w, line_curr_h = font.getbbox(line_curr)
        if line_curr_w + word_w < col_w - col_gap:
            line_curr += f'{word} '
        else:
            lines.append(line_curr.strip())
            line_curr = f'{word} '
    lines.append(line_curr)
    line_i = 0
    for line in lines:
        x_curr = text_area_x1
        y_line = text_area_y1 + font_size*line_i*line_spacing
        draw.text((x_curr, y_line), line, '#000000', font=font)
        line_i += 1
    y_curr = text_area_y1 + body_font_size*2*len(lines) + line_h

    ## definition
    text = data['definition']
    font_size = body_font_size
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    lines = []
    line_curr = ''
    for word in text.split(' '):
        _, _, word_w, word_h = font.getbbox(word)
        _, _, line_curr_w, line_curr_h = font.getbbox(line_curr)
        if line_curr_w + word_w < col_w - col_gap:
            line_curr += f'{word} '
        else:
            lines.append(line_curr.strip())
            line_curr = f'{word} '
    lines.append(line_curr)
    line_height = 1.2
    line_i = 0
    for line in lines:
        x_curr = text_area_x1
        y_line = y_curr + body_font_size*line_i*line_spacing
        draw.text((x_curr, y_line), line, '#000000', font=font)
        line_i += 1
    y_curr = text_area_y1

    ## image
    plant_name_scientific = data['plants'][0]['plant_name_scientific']
    plant_slug = plant_name_scientific.lower().strip().replace(' ', '-')
    plant_image_filepath_out = f'images/{page_i}-{ailment_slug}-{plant_slug}.jpg'
    if not os.path.exists(plant_image_filepath_out): 
        prompt = f'''
            {plant_name_scientific},
            outdoor,
            natural light,
            depth of field, bokeh, 
            high resolution, cinematic
        '''
        image = pipe(prompt=prompt, num_inference_steps=25).images[0]
        image.save(plant_image_filepath_out)
    x1 = int(text_area_x1 + col_w*0)
    y1 = int(text_area_y1 + row_h*1)
    plant_img_w = int(col_w - col_gap)
    plant_img_h = int(row_h - row_gap)
    foreground = Image.open(plant_image_filepath_out)
    foreground = img_resize(foreground, plant_img_w, plant_img_h)
    img.paste(foreground, (x1, y1))

    ## causes
    x_start = text_area_x1 + col_w*0
    y_start = text_area_y1 + row_h*2

    y_curr = y_start
    line = 'causes'.upper()
    font_size = body_font_size
    font = ImageFont.truetype(font_helvetica_bold_filepath, font_size)
    draw.text((x_start, y_curr), line, '#000000', font=font)
    y_curr += font_size
    y_curr += line_h

    objs = data['causes']
    font_size = body_font_size
    ellipse_size = 12
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    i = 0
    for obj in objs[:10]:
        name = obj['name']
        mentions = str(obj['mentions'])
        x1 = x_start + 32
        y1 = y_curr + line_h*i
        x2 = x1 + ellipse_size
        y2 = y1 + ellipse_size
        draw.ellipse((x1, y1 + ellipse_size, x2, y2 + ellipse_size), fill='black')
        draw.text((x1 + 32, y1), name.capitalize(), '#000000', font=font)
        # _, _, mentions_w, _ = font.getbbox(mentions)
        # draw.text((x1 + col_w - col_gap - mentions_w, y1), mentions, '#000000', font=font)
        i += 1

    ## symptoms
    x_start = text_area_x1 + col_w*0
    y_start = text_area_y1 + row_h*3

    y_curr = y_start
    line = 'symptoms'.upper()
    font_size = body_font_size
    font = ImageFont.truetype(font_helvetica_bold_filepath, font_size)
    draw.text((x_start, y_curr), line, '#000000', font=font)
    y_curr += font_size
    y_curr += line_h

    objs = data['symptoms']
    font_size = body_font_size
    ellipse_size = 12
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    i = 0
    for obj in objs[:10]:
        name = obj['name']
        mentions = str(obj['mentions'])
        x1 = x_start + 32
        y1 = y_curr + line_h*i
        x2 = x1 + ellipse_size
        y2 = y1 + ellipse_size
        draw.ellipse((x1, y1 + ellipse_size, x2, y2 + ellipse_size), fill='black')
        draw.text((x1 + 32, y1), name.capitalize(), '#000000', font=font)
        # _, _, mentions_w, _ = font.getbbox(mentions)
        # draw.text((x1 + col_w - col_gap - mentions_w, y1), mentions, '#000000', font=font)
        i += 1

    ## plants
    x_start = text_area_x1 + col_w*1
    y_start = text_area_y1 + row_h*0

    y_curr = y_start
    line = 'medicinal plants'.upper()
    font_size = body_font_size
    font = ImageFont.truetype(font_helvetica_bold_filepath, font_size)
    draw.text((x_start, y_curr), line, '#000000', font=font)
    y_curr += font_size
    y_curr += line_h

    objs = data['plants']
    font_size = body_font_size
    ellipse_size = 12
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    i = 0
    for obj in objs[:10]:
        name = obj['plant_name_scientific']
        mentions = str(obj['plant_mentions'])
        x1 = x_start + 32
        y1 = y_curr + line_h*i
        x2 = x1 + ellipse_size
        y2 = y1 + ellipse_size
        draw.ellipse((x1, y1 + ellipse_size, x2, y2 + ellipse_size), fill='black')
        draw.text((x1 + 32, y1), name, '#000000', font=font)
        # _, _, mentions_w, _ = font.getbbox(mentions)
        # draw.text((x1 + col_w - col_gap - mentions_w, y1), mentions, '#000000', font=font)
        i += 1

    ## page num
    font_size = 32
    line = str(page_i)
    font = ImageFont.truetype(font_helvetica_regular_filepath, font_size)
    _, _, line_w, _ = font.getbbox(line)
    x1 = a4_w - 3*cell_size - line_w - cell_size//2
    draw.text((x1, 40*cell_size), line, '#000000', font=font)

    img.save(image_filepath_out)
    img.show()
    quit()


if 0:
    ## gen images
    checkpoint_filepath = '/home/ubuntu/vault/stable-diffusion/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors'
    pipe = StableDiffusionXLPipeline.from_single_file(
        checkpoint_filepath, 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
    ).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    page_i = 0
    for filename in os.listdir(jsons_folderpath):
        json_filepath = f'{jsons_folderpath}/{filename}'
        print('***********************')
        print(json_filepath)
        print('***********************')
        draw_ailment_image(json_filepath, page_i)
        # draw_ailment_plants_images(json_filepath, page_i+1)
        page_i += 1
else:
    ## gen texts
    page_i = 0
    for filename in os.listdir(jsons_folderpath):
        json_filepath = f'{jsons_folderpath}/{filename}'
        print('***********************')
        print(json_filepath)
        print('***********************')

        if page_i % 2 == 0:
            side = 'r'
        else:
            side = 'l'
        # draw_page_ailment(json_filepath, page_i, regen=True)
        draw_page(json_filepath, page_i, side, regen=True)
        # draw_page_herbs(json_filepath, page_i+1, regen=True)
        # preview_full(json_filepath, page_i)
        page_i += 1
        # break

