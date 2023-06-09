import jinja2

def main():
    HEIGHT = 1123
    WIDTH = 794
    SIZE = 20.04
    STARTX = 0.25
    STARTY = 0.25
    tpl = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg" width="{{WIDTH}}" height="{{HEIGHT}}">
            <style>
                line{
                    stroke:black;
                    stroke-opacity:0.25;
                    stroke-width: 0.5;
                    shape-rendering: crispEdges;
                }
            </style>
        {{ vertical_lines }}
        {{ horizontal_lines }}
    </svg>
    """
    vert = [f'<line x1="{STARTX + (x*SIZE)}" y1="0" x2="{STARTX + (x*SIZE)}" y2="3580"/>' for x in range(int(WIDTH/SIZE)+5)]
    vert = '\n'.join(vert)
    
    hori = [f'<line x1="0" y1="{STARTY + (y*SIZE)}" x2="2480" y2="{STARTY + (y*SIZE)}"/>' for y in range(int(HEIGHT/SIZE)+5)]
    hori = '\n'.join(hori)
    temp = jinja2.Template(tpl)
    svg = temp.render(WIDTH=WIDTH,HEIGHT=HEIGHT,vertical_lines=vert,horizontal_lines=hori)
    with open('assets/grid.svg','w') as f:
        f.write(svg)

if __name__ == "__main__":
    main()