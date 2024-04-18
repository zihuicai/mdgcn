import time


output_file_path = "out.txt"
out_to_file = True
out_to_console = True

# print, and it can display the print time
def tprint(text, display_time=False, end='\n'):
    text = str(text)
    time_str = '[' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '] '
    space_str = ' ' * len(time_str)
    word_cells, cells = text.split('\n'), []
    for word_cell in word_cells:
        cells.append(space_str + word_cell)
        cells.append('\n')
    del cells[-1]
    if display_time:
        cells[0] = cells[0].replace(space_str, time_str, 1)
    cells.append(end)
    if out_to_file:
        with open(output_file_path, 'a') as f:
            for cell in cells:
                f.writelines(cell)
    if out_to_console:
        for cell in cells:
            print(cell, end='')

