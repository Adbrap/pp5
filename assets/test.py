import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


def on_close(event):
    print('Closed Figure!')

fig = plt.figure()
fig.canvas.mpl_connect('close_event', on_close)
at = AnchoredText(
    "Figure 1a", prop=dict(size=15), frameon=True, loc='upper left')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.text(0.35, 0.5, 'Close Me!', dict(size=30))
plt.show()