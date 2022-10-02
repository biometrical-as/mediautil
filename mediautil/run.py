import typer 
from typer import Argument, Option

from mediautil import Vid, Player

app = typer.Typer() 

@app.command()
def main(
    source: str = Argument(..., show_default=False, help='Video source'), 
    fps: int = Option(30, min=1, help='FPS')
): 
    vid = Vid(source)
    player = Player(fps, name='Mediautil player')

    for frame in vid: 
        player.add_frame(frame)
    
    player.wait()

if __name__ == '__main__': 
    app()