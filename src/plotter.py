import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from solvers import Solver


# Line style for the grid. Options are:
# | type             | short form | long form
# | solid line       | '-'        | 'solid'
# | dashed line      | '--'       | 'dashed'
# | dash-dotted line | '-.'       | 'dashdot'
# | dotted line      | ':'        | 'dotted'
# | draw nothing     | 'None'     | ' ' or ''
LINE_STYLE = 'dashed'
ALPHA = 0.3


def plot(solvers, outfile):
    """
    Plot the results by multi-armed bandit solvers.
        solvers (list<Solver>): All of them should have been fitted.
        outfile (str): name of output file
    """
    matplotlib.use('Agg')

    assert all(isinstance(s, Solver) for s in solvers)
    assert all(len(s.regrets) > 0 for s in solvers)

    b = solvers[0].bandit

    # Use a 14 in width and 4 in height
    fig = plt.figure(figsize=(14, 4))

    # Adjust subplot spacing
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    # Create subplots with 1 row and 3 columns (hence the 13 prefix)
    # Then, increment the column index, which start at one
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Plot 1. Cumulative regrets over time
    for s in solvers:
        ax1.plot(s.regrets, label=s.name)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cumulative Regret')
    ax1.legend(loc='upper center', bbox_to_anchor=(1.82, -0.25), ncol=len(solvers))
    ax1.grid(linestyle=LINE_STYLE, alpha=ALPHA)

    # Plot 2. Probabilities estimated by solvers
    sorted_indexes = sorted(range(b.n), key=lambda x: b.probas[x])
    ax2.plot(
        [b.probas[x] for x in sorted_indexes],
        color='black',
        linestyle=LINE_STYLE,
        markersize=12,
    )
    for s in solvers:
        ax2.plot(
            [s.estimated_probas[x] for x in sorted_indexes],
            'x',
            markeredgewidth=2,
            label=s.name,
        )
    ax2.set_xlabel('Actions sorted by reward probability')
    ax2.set_ylabel('Actual / Estimated reward probability')
    ax2.grid(linestyle=LINE_STYLE, alpha=ALPHA)

    # Plot 3: Action counts
    # TODO Hard to tell which action corresponds to the step in the graph
    for s in solvers:
        ax3.plot(
            np.array(s.counts) / len(solvers[0].regrets),
            drawstyle='steps',
            linewidth=2,
        )
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('No. of trials')
    ax3.grid(linestyle=LINE_STYLE, alpha=ALPHA)

    plt.savefig(outfile)
