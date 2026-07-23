import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon
from mpl_toolkits.mplot3d import Axes3D
from sympy import lambdify, Symbol, sympify
import inspect

GRAPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

COLORS = ['#4CAF50', '#2196F3', '#F44336', '#FF9800', '#9C27B0', '#00BCD4', '#FFC107']


def make_graph(expr_or_data, var='x', title="Graph", graph_type="2d", **kwargs):
    filename = f"graph_{int(np.random.random()*1e6)}.png"
    filepath = os.path.join(GRAPH_DIR, filename)

    if graph_type == "2d" and isinstance(expr_or_data, list):
        graph_type = "multi"

    if graph_type == "2d":
        _plot_2d(expr_or_data, var, title, filepath, **kwargs)
    elif graph_type == "3d":
        _plot_3d(expr_or_data, var, title, filepath, **kwargs)
    elif graph_type == "multi":
        _plot_multi(expr_or_data, title, filepath, **kwargs)
    elif graph_type == "geometry":
        _plot_geometry(expr_or_data, title, filepath, **kwargs)
    else:
        _plot_2d(expr_or_data, var, title, filepath, **kwargs)

    return filepath


def _plot_2d(expr, var, title, filepath, x_range=(-10, 10), points=500, **kwargs):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    x_vals = np.linspace(x_range[0], x_range[1], points)

    if isinstance(expr, str):
        sym = Symbol(var) if isinstance(var, str) else var
        expr = sympify(expr)
        f = lambdify(sym, expr, "numpy")
        y_vals = f(x_vals)
    elif callable(expr):
        y_vals = expr(x_vals)
    else:
        y_vals = expr

    ax.plot(x_vals, y_vals, color=COLORS[0], linewidth=2, label=str(expr) if not callable(expr) else title)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(var)
    ax.set_ylabel('f(%s)' % var)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    return filepath


def _plot_3d(expr, var, title, filepath, x_range=(-5, 5), y_range=(-5, 5), points=50, **kwargs):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    x_vals = np.linspace(x_range[0], x_range[1], points)
    y_vals = np.linspace(y_range[0], y_range[1], points)
    X, Y = np.meshgrid(x_vals, y_vals)

    if isinstance(expr, str):
        from sympy import symbols, sympify
        xs, ys = symbols('x y')
        e = sympify(expr)
        f = lambdify((xs, ys), e, "numpy")
        Z = f(X, Y)
    elif callable(expr):
        Z = expr(X, Y)
    else:
        Z = expr

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    return filepath


def _plot_multi(exprs, title, filepath, var='x', x_range=(-10, 10), points=500, **kwargs):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    x_vals = np.linspace(x_range[0], x_range[1], points)

    for i, e in enumerate(exprs):
        color = COLORS[i % len(COLORS)]
        if isinstance(e, str):
            sym = Symbol(var) if isinstance(var, str) else var
            expr = sympify(e)
            f = lambdify(sym, expr, "numpy")
            y_vals = f(x_vals)
            ax.plot(x_vals, y_vals, color=color, linewidth=2, label=e)
        elif callable(e):
            y_vals = e(x_vals)
            ax.plot(x_vals, y_vals, color=color, linewidth=2, label=f"fn{i}")
        else:
            ax.plot(x_vals, e, color=color, linewidth=2, label=f"series{i}")

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    return filepath


def _plot_geometry(shapes, title, filepath, **kwargs):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    for shape in shapes:
        if shape.get('type') == 'circle':
            c = Circle((shape['center'][0], shape['center'][1]), shape['radius'],
                       fill=False, edgecolor=COLORS[0], linewidth=2)
            ax.add_patch(c)
        elif shape.get('type') == 'polygon':
            pts = np.array(shape['points'])
            poly = MplPolygon(pts, fill=False, edgecolor=COLORS[1], linewidth=2)
            ax.add_patch(poly)
        elif shape.get('type') == 'line':
            xs = [shape['start'][0], shape['end'][0]]
            ys = [shape['start'][1], shape['end'][1]]
            ax.plot(xs, ys, color=COLORS[2], linewidth=2)
        elif shape.get('type') == 'point':
            ax.plot(shape['pos'][0], shape['pos'][1], 'o', color=COLORS[3], markersize=8)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    return filepath