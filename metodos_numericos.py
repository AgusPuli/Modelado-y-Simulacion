#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicación de Métodos Numéricos - Modelado y Simulación
Basado en: "Fundamentos de Modelado y Simulación" - Omar J. Cáceres, 2ª Ed. 2026
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
import re, warnings, random
warnings.filterwarnings('ignore')

# ─── Palette ──────────────────────────────────────────────────────────────────
BG      = "#f0f4f8"
PANEL   = "#ffffff"
HDR     = "#1e3a5f"
HDR_FG  = "#ffffff"
ACCENT  = "#2a6ebb"
ENTRY   = "#eef2f7"
ERR     = "#cc0000"
OK      = "#006600"
FF      = "Segoe UI"

# ─── Math helpers ─────────────────────────────────────────────────────────────
_TR = standard_transformations + (implicit_multiplication_application, convert_xor)

_LOCAL = {
    'e': sp.E, 'pi': sp.pi, 'oo': sp.oo, 'inf': sp.oo,
    'sqrt': sp.sqrt, 'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
    'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
    'abs': sp.Abs, 'Abs': sp.Abs, 'E': sp.E,
}

def parse_math(s, extra=None):
    if not s or not s.strip():
        raise ValueError("Expresión vacía")
    s = s.strip()
    s = re.sub(r'\bln\s*\(', 'log(', s)
    s = re.sub(r'\bsen\s*\(', 'sin(', s)
    s = re.sub(r'\btg\s*\(', 'tan(', s)
    s = re.sub(r'√', 'sqrt', s)
    s = re.sub(r'π', 'pi', s)
    s = re.sub(r'∞', 'oo', s)
    loc = dict(_LOCAL)
    if extra:
        loc.update(extra)
    return parse_expr(s, transformations=_TR, local_dict=loc)

def to_f(expr, var='x'):
    return sp.lambdify(sp.Symbol(var), expr, modules=['numpy', {'Abs': np.abs, 'sign': np.sign}])

def to_f_safe(expr, var='x'):
    """Como to_f pero reemplaza NaN/Inf con el límite simbólico (singularidades removibles)."""
    sym = sp.Symbol(var)
    f_np = sp.lambdify(sym, expr, modules=['numpy', {'Abs': np.abs, 'sign': np.sign}])

    def _scalar(xv):
        try:
            v = float(f_np(xv))
            if np.isfinite(v):
                return v
            lv = sp.limit(expr, sym, xv)
            lf = float(complex(lv.evalf()).real)
            return lf if np.isfinite(lf) else v
        except:
            return float('nan')

    def _safe(xv):
        if np.isscalar(xv):
            return _scalar(float(xv))
        arr = np.asarray(xv, dtype=float)
        try:
            res = np.asarray(f_np(arr), dtype=complex)
            res = np.real(res).astype(float)
        except:
            return np.array([_scalar(x) for x in arr.ravel()], dtype=float).reshape(arr.shape)
        bad = np.where(~np.isfinite(res.ravel()))[0]
        if len(bad):
            rf = res.ravel().copy()
            af = arr.ravel()
            for i in bad:
                rf[i] = _scalar(float(af[i]))
            res = rf.reshape(arr.shape)
        return res

    return _safe

def to_f2(expr, v1='x', v2='y'):
    return sp.lambdify((sp.Symbol(v1), sp.Symbol(v2)), expr,
                       modules=['numpy', {'Abs': np.abs}])

def fval(s):
    try:
        return float(parse_math(s))
    except:
        return float(s)

def validate_float(s, allow_neg=True, allow_zero=True):
    try:
        v = fval(s)
        if not allow_neg and v < 0:
            return False, "Debe ser ≥ 0"
        if not allow_zero and v == 0:
            return False, "No puede ser 0"
        return True, v
    except:
        return False, "Valor inválido"

def validate_int(s, min_val=1):
    try:
        v = int(float(s))
        if v < min_val:
            return False, f"Debe ser ≥ {min_val}"
        return True, v
    except:
        return False, "Entero inválido"

# ─── Scrollable frame ─────────────────────────────────────────────────────────
class ScrollFrame(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=kw.get('bg', PANEL))
        self.canvas = tk.Canvas(self, bg=kw.get('bg', PANEL), highlightthickness=0)
        sb = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.inner = tk.Frame(self.canvas, bg=kw.get('bg', PANEL))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.inner.bind('<Configure>', self._on_configure)
        self.canvas.bind('<Configure>', self._on_canvas)
        self.canvas.bind_all('<MouseWheel>', self._on_wheel)

    def _on_configure(self, e):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas(self, e):
        self.canvas.itemconfig(self._win, width=e.width)

    def _on_wheel(self, e):
        self.canvas.yview_scroll(int(-1*(e.delta/120)), 'units')

# ─── Shared tooltip (one window reused globally) ─────────────────────────────
class _Tooltip:
    """Single shared tooltip window — appears after 600 ms, hides on leave."""
    _win = None
    _after_id = None

    @classmethod
    def schedule(cls, widget, text):
        cls.cancel()
        cls._after_id = widget.after(600, lambda: cls._show(widget, text))

    @classmethod
    def cancel(cls):
        if cls._after_id is not None:
            try:
                # after IDs are widget-relative; we just suppress errors
                pass
            except Exception:
                pass
            cls._after_id = None
        cls.hide()

    @classmethod
    def _show(cls, widget, text):
        cls.hide()
        try:
            x = widget.winfo_rootx() + widget.winfo_width() + 4
            y = widget.winfo_rooty()
            cls._win = tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)
            tw.wm_attributes('-topmost', True)
            tw.wm_geometry(f"+{x}+{y}")
            tk.Label(tw, text=text, bg='#fffbe6', fg='#333',
                     font=(FF, 8), relief='solid', bd=1,
                     wraplength=220, justify='left',
                     padx=6, pady=4).pack()
        except Exception:
            cls._win = None

    @classmethod
    def hide(cls):
        if cls._win is not None:
            try:
                cls._win.destroy()
            except Exception:
                pass
            cls._win = None


# ─── Labelled entry ───────────────────────────────────────────────────────────
class LEntry(tk.Frame):
    def __init__(self, parent, label, default='', width=22, tooltip='', **kw):
        super().__init__(parent, bg=kw.get('bg', PANEL))
        tk.Label(self, text=label, bg=kw.get('bg', PANEL), font=(FF, 9),
                 anchor='w', width=22).pack(side='left')
        self.var = tk.StringVar(value=default)
        self.entry = tk.Entry(self, textvariable=self.var, width=width,
                              bg=ENTRY, relief='flat', font=(FF, 9))
        self.entry.pack(side='left', padx=2)
        self.status = tk.Label(self, text='', bg=kw.get('bg', PANEL),
                               font=(FF, 8), width=3)
        self.status.pack(side='left')
        if tooltip:
            self.entry.bind('<Enter>', lambda e, t=tooltip: _Tooltip.schedule(self.entry, t))
            self.entry.bind('<Leave>', lambda e: _Tooltip.hide())
            self.entry.bind('<FocusOut>', lambda e: _Tooltip.hide())

    def get(self):
        return self.var.get()

    def set_status(self, ok):
        self.status.config(text='✓' if ok else '✗',
                           fg=OK if ok else ERR)

# ─── Treeview table ───────────────────────────────────────────────────────────
def make_table(parent, cols, widths=None):
    frame = tk.Frame(parent, bg=PANEL)
    sb_y = ttk.Scrollbar(frame, orient='vertical')
    sb_x = ttk.Scrollbar(frame, orient='horizontal')
    tree = ttk.Treeview(frame, columns=cols, show='headings',
                        yscrollcommand=sb_y.set, xscrollcommand=sb_x.set,
                        height=8)
    sb_y.config(command=tree.yview)
    sb_x.config(command=tree.xview)
    for i, c in enumerate(cols):
        w = widths[i] if widths else 90
        tree.heading(c, text=c)
        tree.column(c, width=w, anchor='center', minwidth=50)
    sb_y.pack(side='right', fill='y')
    sb_x.pack(side='bottom', fill='x')
    tree.pack(fill='both', expand=True)
    frame.pack(fill='both', expand=True, pady=4)
    return tree

def table_clear(tree):
    for row in tree.get_children():
        tree.delete(row)

def table_insert(tree, values, tag=''):
    vals = [f"{v:.8g}" if isinstance(v, float) else str(v) for v in values]
    tree.insert('', 'end', values=vals, tags=(tag,))
    tree.tag_configure('alt', background='#f0f6ff')

# ─── Section label ────────────────────────────────────────────────────────────
def section(parent, text, bg=PANEL):
    tk.Label(parent, text=text, bg=bg, fg=ACCENT,
             font=(FF, 9, 'bold')).pack(anchor='w', padx=6, pady=(8, 2))
    ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=6)

def result_label(parent, bg=PANEL):
    lbl = tk.Label(parent, text='', bg=bg, fg=ACCENT,
                   font=(FF, 10, 'bold'), wraplength=400, justify='left')
    lbl.pack(anchor='w', padx=6, pady=4)
    return lbl

def theory_box(parent, text, bg=PANEL):  # kept for calculadora tab
    box = tk.Text(parent, height=6, wrap='word', bg='#f7faff',
                  font=(FF, 8), relief='flat', bd=1, state='normal')
    box.insert('1.0', text)
    box.config(state='disabled')
    box.pack(fill='x', padx=6, pady=4)
    return box

def formula_card(parent, formula_lines, desc, bg=PANEL):
    """Renderiza fórmulas con mathtext de matplotlib + descripción textual."""
    card = tk.Frame(parent, bg='#eef3fa', bd=1, relief='solid')
    card.pack(fill='x', padx=6, pady=(4, 2))

    n = max(len(formula_lines), 1)
    fh = 0.42 + 0.54 * n
    fig = Figure(figsize=(3.85, fh), dpi=96, facecolor='#f8f6ff')
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_axis_off()
    ax.set_facecolor('#f8f6ff')
    step = 1.0 / (n + 0.3)
    for i, line in enumerate(formula_lines):
        y = 1.0 - step * (i + 0.65)
        ax.text(0.5, y, line, transform=ax.transAxes,
                ha='center', va='center', fontsize=12.5,
                fontfamily='DejaVu Serif')
    cv = FigureCanvasTkAgg(fig, master=card)
    cv.draw()
    cv.get_tk_widget().pack(fill='x', padx=1, pady=(2, 0))

    box = tk.Text(card, height=5, wrap='word', bg='#ddeeff',
                  font=(FF, 8), relief='flat', bd=0,
                  padx=6, pady=4)
    box.insert('1.0', desc)
    box.config(state='disabled')
    box.pack(fill='x', padx=1, pady=(0, 2))
    return card

def btn(parent, text, cmd, bg=PANEL):
    b = tk.Button(parent, text=text, command=cmd, bg=ACCENT, fg='white',
                  font=(FF, 9, 'bold'), relief='flat', cursor='hand2',
                  activebackground='#1a4d8f', activeforeground='white', padx=10, pady=4)
    b.pack(pady=6, padx=6, anchor='w')
    return b

# ─── Base tab layout ─────────────────────────────────────────────────────────
class TabBase(tk.Frame):
    def __init__(self, parent, title=''):
        super().__init__(parent, bg=BG)
        # Header
        hdr = tk.Frame(self, bg=HDR, height=36)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text=title, bg=HDR, fg=HDR_FG,
                 font=(FF, 11, 'bold')).pack(side='left', padx=12, pady=6)
        # Body PanedWindow
        pw = tk.PanedWindow(self, orient='horizontal', bg=BG,
                            sashrelief='flat', sashwidth=4)
        pw.pack(fill='both', expand=True, padx=4, pady=4)
        # Left: scroll frame
        self.left = ScrollFrame(pw, bg=PANEL)
        pw.add(self.left, minsize=340, width=400)
        # Right: plot frame
        self.right = tk.Frame(pw, bg=PANEL)
        pw.add(self.right, minsize=400)
        self.fig = Figure(figsize=(5, 4), dpi=96, facecolor=PANEL)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas_fig.get_tk_widget().pack(fill='both', expand=True)
        tb = NavigationToolbar2Tk(self.canvas_fig, self.right)
        tb.update()
        self.lp = self.left.inner  # shortcut

    def draw(self):
        self.canvas_fig.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – BISECCIÓN
# ══════════════════════════════════════════════════════════════════════════════
class TabBiseccion(TabBase):
    _FLINES = [
        r'$c_n = \dfrac{a_n + b_n}{2}$',
        r'$|E_n| \leq \dfrac{b - a}{2^{n+1}}$',
        r'$n_{\min} \geq \log_2\left(\dfrac{b-a}{\varepsilon}\right) - 1$',
    ]
    _DESC = (
        "Halla raíces de f(x)=0 dividiendo el intervalo a la mitad (Th. Bolzano: f(a)·f(b)<0).\n\n"
        "▸ a, b: extremos del intervalo  ▸ c: punto medio = raíz estimada\n"
        "▸ f(c): residuo en c  ▸ Error: cota máxima = (b−a)/2\n\n"
        "✦ Convergencia lineal garantizada. ~20 iter para tol=1e-6 en [0,1]. "
        "El error se reduce exactamente a la mitad en cada paso."
    )

    def __init__(self, parent):
        super().__init__(parent, '📐 Bisección — Búsqueda Binaria de Raíces')
        lp = self.lp
        formula_card(lp, self._FLINES, self._DESC)
        section(lp, 'Parámetros')
        self.ef   = LEntry(lp, 'f(x) =', 'x**3 - x - 2', tooltip='Ej: x^3-x-2, sin(x)-x/2')
        self.ea   = LEntry(lp, 'a (límite inf) =', '1')
        self.eb   = LEntry(lp, 'b (límite sup) =', '2')
        self.etol = LEntry(lp, 'Tolerancia (ε) =', '1e-6', tooltip='Error máximo permitido')
        self.eit  = LEntry(lp, 'Máx. iteraciones =', '100')
        for w in [self.ef, self.ea, self.eb, self.etol, self.eit]:
            w.pack(fill='x', padx=6, pady=1)
        btn(lp, '▶  Calcular', self._run)
        section(lp, 'Resultados por iteración')
        self.tree = make_table(lp,
            ['n', 'a', 'b', 'c', 'f(c)', 'Error'],
            [40, 100, 100, 100, 100, 100])
        self.res_lbl = result_label(lp)

    def _run(self):
        try:
            expr = parse_math(self.ef.get())
            f = to_f(expr)
            a = fval(self.ea.get())
            b = fval(self.eb.get())
            tol = fval(self.etol.get())
            ok, N = validate_int(self.eit.get(), 1)
            if not ok:
                raise ValueError(N)
            if tol <= 0:
                raise ValueError("Tolerancia debe ser > 0")
            if a >= b:
                raise ValueError("Debe ser a < b")
            fa, fb = float(f(a)), float(f(b))
            # If endpoint is already a root
            if abs(fa) < tol:
                self.res_lbl.config(text=f"✓ Raíz = a = {a:.10g}  |  f(a) = {fa:.4e}  |  Iteraciones: 0", fg=OK)
                self._plot(expr, a, b, a, [], [])
                return
            if abs(fb) < tol:
                self.res_lbl.config(text=f"✓ Raíz = b = {b:.10g}  |  f(b) = {fb:.4e}  |  Iteraciones: 0", fg=OK)
                self._plot(expr, a, b, b, [], [])
                return
            if fa * fb > 0:
                raise ValueError("f(a) y f(b) deben tener signos opuestos (o un extremo es raíz)")

            table_clear(self.tree)
            rows = []
            iters_err = []
            for i in range(N):
                c = (a + b) / 2
                fc = float(f(c))
                err = (b - a) / 2
                rows.append((i+1, a, b, c, fc, err))
                iters_err.append((i+1, err))
                if abs(fc) < tol or err < tol:
                    table_insert(self.tree, (i+1, round(a,8), round(b,8),
                                             round(c,8), round(fc,8), round(err,8)),
                                 'alt' if i%2 else '')
                    self._plot(expr, a, b, c, rows, iters_err)
                    self.res_lbl.config(
                        text=f"✓ Raíz ≈ {c:.10g}  |  f(c) = {fc:.4e}  |  Iteraciones: {i+1}",
                        fg=OK)
                    return
                table_insert(self.tree, (i+1, round(a,8), round(b,8),
                                         round(c,8), round(fc,8), round(err,8)),
                             'alt' if i%2 else '')
                if fa * fc < 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc
            self.res_lbl.config(text="⚠ No convergió en el máximo de iteraciones.", fg=ERR)
            self._plot(expr, a, b, c, rows, iters_err)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _plot(self, expr, a0, b0, c, rows, iters_err):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        f = to_f(expr)
        margin = max(0.5, (b0 - a0))
        xs = np.linspace(a0 - margin, b0 + margin, 400)
        try:
            ys = f(xs)
            ax1.plot(xs, ys, 'b-', lw=2, label='f(x)')
        except:
            pass
        ax1.axhline(0, color='k', lw=0.7)
        ax1.axvline(c, color='r', ls='--', lw=1.2, label=f'c={c:.5g}')
        ax1.scatter([c], [float(f(c))], color='red', zorder=5, s=60)
        ax1.set_title('Función y raíz', fontsize=9)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        if iters_err:
            ns, es = zip(*iters_err)
            ax2.semilogy(ns, es, 'o-', color=ACCENT, lw=1.5, ms=4)
            ax2.set_title('Error por iteración', fontsize=9)
            ax2.set_xlabel('Iteración')
            ax2.set_ylabel('Error')
            ax2.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – PUNTO FIJO
# ══════════════════════════════════════════════════════════════════════════════
class TabPuntoFijo(TabBase):
    _FLINES = [
        r'$x_{n+1} = g(x_n)$',
        r'Convergencia $\Leftarrow\; |g\'(x_0)| < 1$  (Banach)',
        r'$|E_n| \approx |x_{n+1} - x_n|$',
    ]
    _DESC = (
        "Reformula f(x)=0 como x=g(x) e itera desde x₀. El punto fijo x* cumple g(x*)=x*.\n\n"
        "▸ g(x): función de iteración (reescribir f=0 como x=g(x))\n"
        "▸ x₀: valor inicial  ▸ Error = |x_{n+1}−x_n|\n\n"
        "✦ Converge si g es contractiva (|g'|<1). Convergencia lineal. "
        "Si |g'(x*)|≈0 converge más rápido. Resultado: f(x*)≈0."
    )

    def __init__(self, parent):
        super().__init__(parent, '🔁 Punto Fijo')
        lp = self.lp
        formula_card(lp, self._FLINES, self._DESC)
        section(lp, 'Parámetros')
        self.ef   = LEntry(lp, 'f(x) (original) =', 'cos(x) - x', tooltip='La ecuación f(x)=0 a resolver')
        self.eg   = LEntry(lp, 'g(x) (iteración) =', 'cos(x)', tooltip='Reescribir f(x)=0 como x=g(x)')
        self.ex0  = LEntry(lp, 'x₀ (valor inicial) =', '1.0')
        self.etol = LEntry(lp, 'Tolerancia (ε) =', '1e-6')
        self.eit  = LEntry(lp, 'Máx. iteraciones =', '100')
        for w in [self.ef, self.eg, self.ex0, self.etol, self.eit]:
            w.pack(fill='x', padx=6, pady=1)
        btn(lp, '▶  Calcular', self._run)
        section(lp, 'Resultados por iteración')
        self.tree = make_table(lp, ['n', 'x_n', 'g(x_n)', 'Error'], [40,110,110,100])
        self.res_lbl = result_label(lp)

    def _run(self):
        try:
            exprf = parse_math(self.ef.get())
            exprg = parse_math(self.eg.get())
            f = to_f(exprf)
            g = to_f(exprg)
            x = fval(self.ex0.get())
            tol = fval(self.etol.get())
            ok, N = validate_int(self.eit.get(), 1)
            if not ok:
                raise ValueError(N)
            # Check contractivity
            x_sym = sp.Symbol('x')
            gp = sp.diff(exprg, x_sym)
            gp_at_x0 = float(gp.subs(x_sym, x).evalf())
            warn_contract = abs(gp_at_x0) >= 1

            table_clear(self.tree)
            iters_err = []
            history = [x]
            for i in range(N):
                gx = float(g(x))
                err = abs(gx - x)
                table_insert(self.tree, (i+1, round(x,8), round(gx,8), round(err,8)),
                             'alt' if i%2 else '')
                iters_err.append((i+1, err))
                history.append(gx)
                if err < tol:
                    msg = f"✓ Punto fijo ≈ {gx:.10g}  |  f(x*)={float(f(gx)):.4e}  |  Iter: {i+1}"
                    if warn_contract:
                        msg += "\n⚠ |g'(x₀)| ≥ 1: convergencia no garantizada teóricamente"
                    self.res_lbl.config(text=msg, fg=OK if not warn_contract else 'orange')
                    self._plot(exprf, exprg, history, iters_err)
                    return
                x = gx
            self.res_lbl.config(text="⚠ No convergió.", fg=ERR)
            self._plot(exprf, exprg, history, iters_err)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _plot(self, exprf, exprg, history, iters_err):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        f = to_f(exprf)
        g = to_f(exprg)
        if len(history) > 1:
            xmin = min(history) - 0.5
            xmax = max(history) + 0.5
        else:
            xmin, xmax = history[0]-2, history[0]+2
        xs = np.linspace(xmin, xmax, 400)
        try:
            ax1.plot(xs, f(xs), 'b-', lw=2, label='f(x)')
        except: pass
        try:
            ax1.plot(xs, g(xs), 'g--', lw=1.5, label='g(x)')
        except: pass
        ax1.plot(xs, xs, 'gray', lw=1, label='y=x')
        ax1.axhline(0, color='k', lw=0.7)
        ax1.scatter(history, [float(g(v)) for v in history], color='red', s=20, zorder=5)
        ax1.set_title('Cobweb y función', fontsize=9)
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        if iters_err:
            ns, es = zip(*iters_err)
            ax2.semilogy(ns, es, 'o-', color=ACCENT, lw=1.5, ms=4)
            ax2.set_title('Error por iteración', fontsize=9)
            ax2.set_xlabel('Iteración'); ax2.set_ylabel('Error')
            ax2.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – AITKEN
# ══════════════════════════════════════════════════════════════════════════════
class TabAitken(TabBase):
    _FLINES = [
        r'$\hat{x}_n = x_n - \dfrac{(x_{n+1} - x_n)^2}{x_{n+2} - 2\,x_{n+1} + x_n}$',
        r'Requiere denom $= x_{n+2}-2x_{n+1}+x_n \neq 0$',
    ]
    _DESC = (
        "Acelera una sucesión convergente usando extrapolación cuadrática de 3 términos consecutivos.\n\n"
        "▸ g(x): función de iteración base  ▸ xₙ, xₙ₊₁, xₙ₊₂: tres iterados\n"
        "▸ x̂ₙ: estimación acelerada  ▸ Error = |x̂ₙ − xₙ|\n\n"
        "✦ Muy útil cuando la iteración base converge lentamente. "
        "Si el denominador → 0, el punto fijo ya fue alcanzado. "
        "Converge más rápido que el método base."
    )

    def __init__(self, parent):
        super().__init__(parent, '⚡ Aceleración de Aitken')
        lp = self.lp
        formula_card(lp, self._FLINES, self._DESC)
        section(lp, 'Parámetros')
        self.eg   = LEntry(lp, 'g(x) (iteración) =', 'cos(x)', tooltip='Función de iteración x=g(x)')
        self.ex0  = LEntry(lp, 'x₀ (valor inicial) =', '1.0')
        self.etol = LEntry(lp, 'Tolerancia (ε) =', '1e-8')
        self.eit  = LEntry(lp, 'Máx. iteraciones =', '50')
        for w in [self.eg, self.ex0, self.etol, self.eit]:
            w.pack(fill='x', padx=6, pady=1)
        btn(lp, '▶  Calcular', self._run)
        section(lp, 'Resultados por iteración')
        self.tree = make_table(lp,
            ['n', 'xₙ', 'xₙ₊₁', 'xₙ₊₂', 'x̂ₙ (Aitken)', 'Error'],
            [35, 95, 95, 95, 95, 85])
        self.res_lbl = result_label(lp)

    def _run(self):
        try:
            exprg = parse_math(self.eg.get())
            g = to_f(exprg)
            x = fval(self.ex0.get())
            tol = fval(self.etol.get())
            ok, N = validate_int(self.eit.get(), 1)
            if not ok: raise ValueError(N)

            table_clear(self.tree)
            history = [x]
            iters_err = []
            converged = False
            for i in range(N):
                x0 = history[-1]
                # g may return complex for fractional powers of negatives — take real part
                raw1 = g(x0)
                raw2 = g(float(np.real(raw1)) if np.iscomplex(raw1) else raw1)
                x1 = float(np.real(raw1))
                x2 = float(np.real(raw2))
                denom = x2 - 2*x1 + x0
                # Denominator ≈ 0 means sequence is already at fixed point
                if abs(denom) < 1e-14:
                    err = abs(x1 - x0)
                    table_insert(self.tree,
                        (i+1, round(x0,8), round(x1,8), round(x2,8), round(x1,8), round(err,8)),
                        'alt' if i%2 else '')
                    if err < tol:
                        self.res_lbl.config(
                            text=f"✓ Solución ≈ {x1:.12g}  |  Iter: {i+1}\n(denominador→0: punto fijo alcanzado)", fg=OK)
                    else:
                        self.res_lbl.config(
                            text=f"⚠ Denominador ≈ 0 en iteración {i+1}. El método no puede continuar.\n"
                                 f"Última estimación: {x1:.10g}", fg='orange')
                    self._plot(exprg, history, iters_err)
                    return
                x_hat = x0 - (x1 - x0)**2 / denom
                err = abs(x_hat - x0)
                table_insert(self.tree,
                    (i+1, round(x0,8), round(x1,8), round(x2,8), round(x_hat,8), round(err,8)),
                    'alt' if i%2 else '')
                iters_err.append((i+1, err))
                history.append(x_hat)
                if err < tol:
                    self.res_lbl.config(
                        text=f"✓ Solución acelerada ≈ {x_hat:.12g}  |  Iter: {i+1}", fg=OK)
                    converged = True
                    break
            if not converged:
                self.res_lbl.config(text="⚠ No convergió en el máximo de iteraciones.", fg=ERR)
            self._plot(exprg, history, iters_err)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _plot(self, exprg, history, iters_err):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        g = to_f(exprg)
        ax1.plot(range(len(history)), history, 'o-', color=ACCENT, lw=1.5, ms=4)
        ax1.set_title('Convergencia de Aitken', fontsize=9)
        ax1.set_xlabel('Iteración'); ax1.set_ylabel('x̂ₙ')
        ax1.grid(True, alpha=0.3)
        if iters_err:
            ns, es = zip(*iters_err)
            ax2.semilogy(ns, es, 'o-', color='red', lw=1.5, ms=4)
            ax2.set_title('Error (log)', fontsize=9)
            ax2.set_xlabel('Iteración'); ax2.set_ylabel('Error')
            ax2.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – NEWTON-RAPHSON
# ══════════════════════════════════════════════════════════════════════════════
class TabNewtonRaphson(TabBase):
    _FLINES = [
        r'$x_{n+1} = x_n - \dfrac{f(x_n)}{f\'(x_n)}$',
        r'$f\'(x_n) \neq 0,\quad$ Error $= |x_{n+1} - x_n|$',
    ]
    _DESC = (
        "Proyecta la tangente a f en xₙ para obtener la siguiente aproximación. "
        "La derivada se calcula simbólicamente (automática).\n\n"
        "▸ xₙ: aproximación actual  ▸ f(xₙ): residuo  ▸ f'(xₙ): pendiente (tangente)\n"
        "▸ xₙ₊₁: nueva raíz estimada  ▸ Error = |xₙ₊₁−xₙ|\n\n"
        "✦ Convergencia cuadrática: los dígitos correctos se duplican por iteración. "
        "Necesita x₀ cercano a la raíz. Falla si f'(xₙ)=0."
    )

    def __init__(self, parent):
        super().__init__(parent, '📈 Newton-Raphson')
        lp = self.lp
        formula_card(lp, self._FLINES, self._DESC)
        section(lp, 'Parámetros')
        self.ef   = LEntry(lp, 'f(x) =', 'x**3 - x - 2', tooltip='Ej: x^3-2x-5')
        self.ex0  = LEntry(lp, 'x₀ (valor inicial) =', '1.5')
        self.etol = LEntry(lp, 'Tolerancia (ε) =', '1e-8')
        self.eit  = LEntry(lp, 'Máx. iteraciones =', '100')
        for w in [self.ef, self.ex0, self.etol, self.eit]:
            w.pack(fill='x', padx=6, pady=1)
        btn(lp, '▶  Calcular', self._run)
        section(lp, 'Resultados por iteración')
        self.tree = make_table(lp,
            ['n', 'xₙ', 'f(xₙ)', "f'(xₙ)", 'xₙ₊₁', 'Error'],
            [35, 95, 95, 95, 95, 80])
        self.res_lbl = result_label(lp)

    def _run(self):
        try:
            expr = parse_math(self.ef.get())
            x_sym = sp.Symbol('x')
            dexpr = sp.diff(expr, x_sym)
            f  = to_f(expr)
            df = to_f(dexpr)
            x  = fval(self.ex0.get())
            tol = fval(self.etol.get())
            ok, N = validate_int(self.eit.get(), 1)
            if not ok: raise ValueError(N)

            table_clear(self.tree)
            iters_err = []
            tangent_pts = []
            for i in range(N):
                fx = float(f(x))
                dfx = float(df(x))
                if abs(dfx) < 1e-15:
                    raise ValueError(f"f'(x) ≈ 0 en x={x}. El método no puede continuar.")
                x_new = x - fx / dfx
                err = abs(x_new - x)
                table_insert(self.tree,
                    (i+1, round(x,8), round(fx,8), round(dfx,8), round(x_new,8), round(err,8)),
                    'alt' if i%2 else '')
                iters_err.append((i+1, err))
                tangent_pts.append((x, fx, dfx))
                if err < tol or abs(fx) < tol:
                    self.res_lbl.config(
                        text=f"✓ Raíz ≈ {x_new:.12g}  |  f(raíz) = {float(f(x_new)):.4e}  |  Iter: {i+1}",
                        fg=OK)
                    self._plot(expr, x_new, tangent_pts, iters_err)
                    return
                x = x_new
            self.res_lbl.config(text="⚠ No convergió.", fg=ERR)
            self._plot(expr, x, tangent_pts, iters_err)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _plot(self, expr, root, tangents, iters_err):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        f = to_f(expr)
        margin = max(1.0, abs(root)*0.5 + 1)
        xs = np.linspace(root - margin, root + margin, 500)
        try:
            ys = f(xs)
            ax1.plot(xs, np.clip(ys, -1e6, 1e6), 'b-', lw=2, label='f(x)')
        except: pass
        ax1.axhline(0, color='k', lw=0.7)
        ax1.axvline(root, color='r', ls='--', lw=1.2, label=f'raíz≈{root:.5g}')
        # Draw last 3 tangents
        for xp, fp, dfp in tangents[-3:]:
            xtan = np.array([xp - 0.4, xp + 0.4])
            ytan = fp + dfp*(xtan - xp)
            ax1.plot(xtan, ytan, 'orange', lw=1, alpha=0.7)
        ax1.scatter([root], [float(f(root))], color='red', zorder=5, s=60)
        ax1.set_title('Newton-Raphson — tangentes', fontsize=9)
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-5, 5)
        if iters_err:
            ns, es = zip(*iters_err)
            ax2.semilogy(ns, es, 'o-', color=ACCENT, lw=1.5, ms=4)
            ax2.set_title('Error (log)', fontsize=9)
            ax2.set_xlabel('Iteración'); ax2.set_ylabel('Error')
            ax2.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 – LAGRANGE
# ══════════════════════════════════════════════════════════════════════════════
class TabLagrange(TabBase):
    _FLINES = [
        r'$P(x) = \sum_{i=0}^{n} y_i \, L_i(x)$',
        r'$L_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$',
        r'$\text{Error} = \frac{f^{(n+1)}(\xi)}{(n+1)!}\,\prod_{i}(x - x_i)$',
    ]
    _DESC = (
        "Interpolación de Lagrange construye el polinomio único de grado ≤ n que pasa "
        "por los (n+1) nodos (xᵢ, yᵢ). Lᵢ(x) son las bases de Lagrange: valen 1 en xᵢ "
        "y 0 en el resto de nodos. El error depende de la (n+1)-ésima derivada de f en "
        "algún ξ del intervalo. Agregue f(x) exacta para ver el error numérico."
    )

    def __init__(self, parent):
        super().__init__(parent, '📊 Interpolación de Lagrange')
        lp = self.lp
        formula_card(lp, self._FLINES, self._DESC)
        section(lp, 'Parámetros')
        self.exs  = LEntry(lp, 'Nodos xᵢ =', '0,1,2,3,4', tooltip='Valores x separados por comas')
        self.eys  = LEntry(lp, 'Valores yᵢ =', '1,2,0,2,3', tooltip='Valores y separados por comas')
        self.ef   = LEntry(lp, 'f(x) exacta (opt.) =', '', tooltip='Función real para comparar error (opcional)')
        self.exv  = LEntry(lp, 'x* a interpolar =', '1.5')
        for w in [self.exs, self.eys, self.ef, self.exv]:
            w.pack(fill='x', padx=6, pady=1)
        btn(lp, '▶  Calcular', self._run)
        section(lp, 'Bases de Lagrange en x*')
        self.tree = make_table(lp, ['i', 'xᵢ', 'yᵢ', 'Lᵢ(x*)', 'yᵢ·Lᵢ(x*)'], [35,80,80,90,90])
        self.res_lbl = result_label(lp)

    def _lagrange(self, xs, ys, xv):
        n = len(xs)
        result = 0.0
        Li_vals = []
        for i in range(n):
            li = 1.0
            for j in range(n):
                if i != j:
                    li *= (xv - xs[j]) / (xs[i] - xs[j])
            Li_vals.append(li)
            result += ys[i] * li
        return result, Li_vals

    def _run(self):
        try:
            xs = [fval(v.strip()) for v in self.exs.get().split(',')]
            ys = [fval(v.strip()) for v in self.eys.get().split(',')]
            if len(xs) != len(ys):
                raise ValueError("xᵢ e yᵢ deben tener la misma cantidad de valores")
            if len(set(xs)) != len(xs):
                raise ValueError("Los nodos xᵢ deben ser distintos")
            xv = fval(self.exv.get())
            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)
            pxv, Li_vals = self._lagrange(xs, ys, xv)

            has_f = bool(self.ef.get().strip())
            f_exact = None
            if has_f:
                expr_f = parse_math(self.ef.get())
                f_exact = to_f_safe(expr_f)

            table_clear(self.tree)
            for i in range(len(xs)):
                table_insert(self.tree,
                    (i, round(xs[i],6), round(ys[i],6),
                     round(Li_vals[i],8), round(ys[i]*Li_vals[i],8)),
                    'alt' if i%2 else '')

            msg = f"P(x*={xv}) = {pxv:.10g}"
            if has_f and f_exact is not None:
                fxv = float(f_exact(xv))
                err = abs(fxv - pxv)
                msg += f"\nf(x*) = {fxv:.10g}  |  Error = {err:.4e}"
            self.res_lbl.config(text=msg, fg=OK)
            self._plot(xs, ys, xv, pxv, f_exact)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _plot(self, xs, ys, xv, pxv, f_exact):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212) if f_exact else None

        margin = max(0.5, (max(xs)-min(xs))*0.2)
        x_plot = np.linspace(min(xs)-margin, max(xs)+margin, 400)
        y_poly = np.array([self._lagrange(xs, ys, xv_)[0] for xv_ in x_plot])
        ax1.plot(x_plot, y_poly, ACCENT+'-', lw=2, label='P(x)')
        if f_exact:
            try:
                yf = f_exact(x_plot)
                ax1.plot(x_plot, yf, 'g--', lw=1.5, label='f(x) exacta')
                if ax2:
                    err_plot = np.abs(yf - y_poly)
                    ax2.semilogy(x_plot, err_plot + 1e-16, 'r-', lw=1.5)
                    ax2.set_title('Error |f(x)−P(x)|', fontsize=9)
                    ax2.set_xlabel('x'); ax2.grid(True, alpha=0.3)
            except: pass
        ax1.scatter(xs, ys, color='red', s=60, zorder=5, label='Puntos')
        ax1.scatter([xv], [pxv], color='orange', s=80, zorder=6, marker='*', label=f'P({xv:.3g})')
        ax1.set_title('Polinomio de Lagrange', fontsize=9)
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 – DIFERENCIAS FINITAS
# ══════════════════════════════════════════════════════════════════════════════
class TabDifFinitas(TabBase):
    _FLINES = [
        r"$f'_{\rm prog}(x_i) \approx \dfrac{f(x_{i+1}) - f(x_i)}{h} \quad O(h)$",
        r"$f'_{\rm cen}(x_i)  \approx \dfrac{f(x_{i+1}) - f(x_{i-1})}{2h} \quad O(h^2)$",
        r"$f''_{\rm cen}(x_i) \approx \dfrac{f(x_{i+1}) - 2f(x_i) + f(x_{i-1})}{h^2}$",
    ]
    _DESC = (
        "Las diferencias finitas aproximan derivadas usando valores de f en puntos vecinos "
        "separados por h. La fórmula progresiva usa f(xᵢ) y f(xᵢ₊₁); la regresiva usa "
        "f(xᵢ₋₁) y f(xᵢ); la central (más precisa, O(h²)) usa ambos extremos. "
        "h pequeño mejora la precisión pero introduce errores de redondeo. "
        "La derivada exacta se calcula simbólicamente para comparar."
    )

    def __init__(self, parent):
        super().__init__(parent, '∂ Diferencias Finitas')
        lp = self.lp
        formula_card(lp, self._FLINES, self._DESC)
        section(lp, 'Parámetros')
        self.ef  = LEntry(lp, 'f(x) =', 'x**3 - 2*x + 1')
        self.eh  = LEntry(lp, 'Paso h =', '0.1', tooltip='h > 0')

        # Mode toggle
        self.mode_var = tk.StringVar(value='single')
        mf = tk.Frame(lp, bg=PANEL); mf.pack(anchor='w', padx=6, pady=(4,0))
        tk.Label(mf, text='Modo:', bg=PANEL, font=(FF,9)).pack(side='left')
        tk.Radiobutton(mf, text='Un punto', variable=self.mode_var, value='single',
                       bg=PANEL, font=(FF,8), command=self._toggle_mode).pack(side='left', padx=4)
        tk.Radiobutton(mf, text='Múltiples puntos / tabla', variable=self.mode_var, value='multi',
                       bg=PANEL, font=(FF,8), command=self._toggle_mode).pack(side='left', padx=4)
        tk.Radiobutton(mf, text='Tabla t / x(t)', variable=self.mode_var, value='data',
                       bg=PANEL, font=(FF,8), command=self._toggle_mode).pack(side='left', padx=4)

        # Single-point frame
        self.frm_single = tk.Frame(lp, bg=PANEL)
        self.ex  = LEntry(self.frm_single, 'Punto x₀ =', '2.0')
        self.ex.pack(fill='x', padx=2, pady=1)
        tk.Label(self.frm_single, text='Rango análisis de error:', bg=PANEL, font=(FF,8)).pack(anchor='w', padx=2)
        self.eh_min = LEntry(self.frm_single, 'h mínimo =', '1e-8')
        self.eh_max = LEntry(self.frm_single, 'h máximo =', '1.0')
        self.eh_min.pack(fill='x', padx=2, pady=1)
        self.eh_max.pack(fill='x', padx=2, pady=1)

        # Multi-point frame
        self.frm_multi = tk.Frame(lp, bg=PANEL)
        self.ex_multi = LEntry(self.frm_multi, 'Puntos xᵢ =', '0,0.1,0.2,0.3,0.4,0.5',
                                tooltip='Valores x separados por comas')
        self.ex_multi.pack(fill='x', padx=2, pady=1)

        # Data table frame (t/x(t) mode)
        self.frm_data = tk.Frame(lp, bg=PANEL)
        self.et_t = LEntry(self.frm_data, 't (valores) =', '0,1,2,3,4,5,6,7,8',
                           tooltip='Tiempos separados por coma')
        self.et_x = LEntry(self.frm_data, 'x(t) (posición) =', '0,1.9,4.2,7.8,12,17,25,32,42',
                           tooltip='Posiciones separadas por coma')
        self.et_t.pack(fill='x', padx=2, pady=1)
        self.et_x.pack(fill='x', padx=2, pady=1)
        tk.Label(self.frm_data, text='Central en puntos interiores. Progresiva/Regresiva en extremos.',
                 bg=PANEL, font=(FF,8), fg='#555', wraplength=320, justify='left').pack(anchor='w', padx=2, pady=2)

        for w in [self.ef, self.eh]:
            w.pack(fill='x', padx=6, pady=1)
        self.frm_single.pack(fill='x', padx=6)
        self.frm_multi.pack_forget()
        self.frm_data.pack_forget()

        btn(lp, '▶  Calcular', self._run)
        section(lp, 'Resultados')
        self.tree = make_table(lp,
            ['xᵢ', 'Progresiva f\'', 'Regresiva f\'', 'Central f\'', 'Exacta f\'', 'Error central',
             'Central f\'\'', 'Exacta f\'\''],
            [65, 90, 90, 90, 90, 80, 90, 90])
        self.tree_data = make_table(lp,
            ['tᵢ', 'x(tᵢ)', 'v = dx/dt', 'a = d²x/dt²'],
            [70, 90, 110, 110])
        self.tree_data.master.pack_forget()
        self.res_lbl = result_label(lp)

    def _toggle_mode(self):
        mode = self.mode_var.get()
        self.frm_single.pack_forget()
        self.frm_multi.pack_forget()
        self.frm_data.pack_forget()
        self.tree.master.pack_forget()
        self.tree_data.master.pack_forget()
        if mode == 'single':
            self.frm_single.pack(fill='x', padx=6)
            self.tree.master.pack(fill='both', expand=True, pady=4)
        elif mode == 'multi':
            self.frm_multi.pack(fill='x', padx=6)
            self.tree.master.pack(fill='both', expand=True, pady=4)
        else:
            self.frm_data.pack(fill='x', padx=6)
            self.tree_data.master.pack(fill='both', expand=True, pady=4)

    def _run(self):
        if self.mode_var.get() == 'data':
            self._run_data()
            return
        try:
            expr = parse_math(self.ef.get())
            x_sym = sp.Symbol('x')
            d1expr = sp.diff(expr, x_sym, 1)
            d2expr = sp.diff(expr, x_sym, 2)
            f   = to_f(expr)
            d1f = to_f(d1expr)
            d2f = to_f(d2expr)
            h   = fval(self.eh.get())
            if h <= 0: raise ValueError("h debe ser > 0")

            if self.mode_var.get() == 'single':
                x0 = fval(self.ex.get())
                pts = [x0]
                hmin = fval(self.eh_min.get())
                hmax = fval(self.eh_max.get())
            else:
                pts = [fval(v.strip()) for v in self.ex_multi.get().split(',')]
                hmin, hmax = 1e-8, 1.0

            table_clear(self.tree)
            for idx, x0 in enumerate(pts):
                fx  = float(f(x0))
                fxp = float(f(x0+h))
                try: fxm = float(f(x0-h))
                except: fxm = None
                try: exact1 = float(d1f(x0))
                except: exact1 = None
                try: exact2 = float(d2f(x0))
                except: exact2 = None

                fwd1 = (fxp - fx) / h
                bwd1 = (fx - fxm) / h if fxm is not None else float('nan')
                cen1 = (fxp - fxm) / (2*h) if fxm is not None else float('nan')
                cen2 = (fxp - 2*fx + fxm) / h**2 if fxm is not None else float('nan')

                err_cen1 = abs(cen1 - exact1) if (exact1 is not None and np.isfinite(cen1)) else '—'
                e2_str   = round(exact2,8) if exact2 is not None else '—'
                c2_str   = round(cen2,8)   if np.isfinite(cen2) else '—'

                table_insert(self.tree,
                    (round(x0,6), round(fwd1,8), round(bwd1,8) if np.isfinite(bwd1) else '—',
                     round(cen1,8) if np.isfinite(cen1) else '—',
                     round(exact1,8) if exact1 is not None else '—',
                     f"{err_cen1:.4e}" if isinstance(err_cen1, float) else err_cen1,
                     c2_str, e2_str),
                    'alt' if idx%2 else '')

            # Summary for single point
            if len(pts) == 1:
                x0 = pts[0]
                e1 = float(d1f(x0)); e2 = float(d2f(x0))
                self.res_lbl.config(
                    text=f"f'(x₀) exacta = {e1:.8g}  |  f''(x₀) exacta = {e2:.8g}", fg=OK)
                self._plot(f, d1f, x0, h, hmin, hmax)
            else:
                self.res_lbl.config(text=f"Tabla calculada para {len(pts)} puntos con h={h}", fg=OK)
                self._plot_multi(f, d1f, pts, h)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _run_data(self):
        try:
            ts = np.array([fval(v.strip()) for v in self.et_t.get().split(',')], dtype=float)
            xs = np.array([fval(v.strip()) for v in self.et_x.get().split(',')], dtype=float)
            if len(ts) != len(xs):
                raise ValueError("t y x deben tener la misma cantidad de valores")
            n = len(ts)
            vs = np.zeros(n)
            accs = np.zeros(n)
            for i in range(n):
                if i == 0:
                    h = ts[1] - ts[0]
                    vs[i] = (xs[1] - xs[0]) / h
                    if n > 2:
                        h1 = ts[1]-ts[0]; h2 = ts[2]-ts[1]
                        accs[i] = 2*(xs[2]*h1 - xs[1]*(h1+h2) + xs[0]*h2) / (h1*h2*(h1+h2))
                    else:
                        accs[i] = float('nan')
                elif i == n-1:
                    h = ts[-1] - ts[-2]
                    vs[i] = (xs[-1] - xs[-2]) / h
                    if n > 2:
                        h1 = ts[-2]-ts[-3]; h2 = ts[-1]-ts[-2]
                        accs[i] = 2*(xs[-3]*h2 - xs[-2]*(h1+h2) + xs[-1]*h1) / (h1*h2*(h1+h2))
                    else:
                        accs[i] = float('nan')
                else:
                    hp = ts[i]-ts[i-1]; hn = ts[i+1]-ts[i]
                    vs[i] = (xs[i+1] - xs[i-1]) / (hp + hn)
                    accs[i] = 2*(xs[i+1]*hp - xs[i]*(hp+hn) + xs[i-1]*hn) / (hp*hn*(hp+hn))
            table_clear(self.tree_data)
            for i in range(n):
                table_insert(self.tree_data,
                    (round(float(ts[i]),4), round(float(xs[i]),4),
                     round(float(vs[i]),6) if np.isfinite(vs[i]) else '—',
                     round(float(accs[i]),6) if np.isfinite(accs[i]) else '—'),
                    'alt' if i%2 else '')
            self.res_lbl.config(text=f"Calculadas velocidad y aceleración para {n} puntos", fg=OK)
            self._plot_data(ts, xs, vs, accs)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _plot_data(self, ts, xs, vs, accs):
        self.fig.clear()
        ax1 = self.fig.add_subplot(311)
        ax2 = self.fig.add_subplot(312)
        ax3 = self.fig.add_subplot(313)
        ax1.plot(ts, xs, 'b-o', lw=1.5, ms=5)
        ax1.set_title('Posición x(t)', fontsize=9)
        ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.grid(True, alpha=0.3)
        fin_v = np.isfinite(vs)
        ax2.plot(ts[fin_v], vs[fin_v], 'g-o', lw=1.5, ms=5)
        ax2.set_title('Velocidad v = dx/dt', fontsize=9)
        ax2.set_xlabel('t'); ax2.set_ylabel('v'); ax2.grid(True, alpha=0.3)
        fin_a = np.isfinite(accs)
        ax3.plot(ts[fin_a], accs[fin_a], 'r-o', lw=1.5, ms=5)
        ax3.set_title('Aceleración a = d²x/dt²', fontsize=9)
        ax3.set_xlabel('t'); ax3.set_ylabel('a'); ax3.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()

    def _plot_multi(self, f, d1f, pts, h):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        margin = max(h, (max(pts)-min(pts))*0.1)
        xs = np.linspace(min(pts)-margin, max(pts)+margin, 300)
        try: ax1.plot(xs, f(xs),   'b-',  lw=2,   label='f(x)')
        except: pass
        try: ax1.plot(xs, d1f(xs), 'g--', lw=1.5, label="f'(x) exacta")
        except: pass
        cen_approx = [(float(f(x+h))-float(f(x-h)))/(2*h) for x in pts]
        exact_vals  = [float(d1f(x)) for x in pts]
        ax1.scatter(pts, cen_approx, color='red',   s=40, zorder=5, label='Central aprox.')
        ax1.scatter(pts, exact_vals, color='orange', s=20, zorder=6, marker='^', label='Exacta')
        ax1.set_title('f(x) y aproximaciones de f\'', fontsize=9)
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        errs = [abs(c-e) for c,e in zip(cen_approx, exact_vals)]
        ax2.bar(pts, errs, width=h*0.6, color=ACCENT, alpha=0.7)
        ax2.set_title('Error absoluto por punto', fontsize=9)
        ax2.set_xlabel('xᵢ'); ax2.set_ylabel('|Error|')
        ax2.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()

    def _plot(self, f, d1f, x0, h, hmin, hmax):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        xs = np.linspace(x0 - 3*h, x0 + 3*h, 300)
        try:
            ax1.plot(xs, f(xs), 'b-', lw=2, label='f(x)')
            ax1.plot(xs, d1f(xs), 'g--', lw=1.5, label="f'(x) exacta")
        except: pass
        ax1.axvline(x0, color='r', ls=':', lw=1, label=f'x₀={x0}')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        ax1.set_title('Función y derivada', fontsize=9)
        # Error vs h
        hs = np.logspace(np.log10(max(hmin, 1e-15)), np.log10(hmax), 60)
        errs_fwd, errs_cen = [], []
        exact1 = float(d1f(x0))
        for hh in hs:
            try:
                fwd = (float(f(x0+hh)) - float(f(x0))) / hh
                cen = (float(f(x0+hh)) - float(f(x0-hh))) / (2*hh)
                errs_fwd.append(abs(fwd - exact1))
                errs_cen.append(abs(cen - exact1))
            except:
                errs_fwd.append(np.nan); errs_cen.append(np.nan)
        ax2.loglog(hs, errs_fwd, 'r-', lw=1.5, label='Progresiva O(h)')
        ax2.loglog(hs, errs_cen, 'b-', lw=1.5, label='Central O(h²)')
        ax2.axvline(h, color='orange', ls='--', label=f'h={h}')
        ax2.set_title('Error vs h', fontsize=9)
        ax2.set_xlabel('h'); ax2.set_ylabel('Error')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  NEWTON-COTES BASE
# ══════════════════════════════════════════════════════════════════════════════
class NCBase(TabBase):
    def __init__(self, parent, title, flines, desc):
        super().__init__(parent, title)
        lp = self.lp
        formula_card(lp, flines, desc)
        section(lp, 'Parámetros')
        self.ef  = LEntry(lp, 'f(x) =', 'sin(x)')
        self.ea  = LEntry(lp, 'a (límite inf) =', '0')
        self.eb  = LEntry(lp, 'b (límite sup) =', 'pi')
        self.en  = LEntry(lp, 'n (subintervalos) =', '10', tooltip=self._n_hint())
        self.ef_ex = LEntry(lp, 'F(x) antiderivada (opt.) =', '', tooltip='Para calcular valor exacto')
        for w in [self.ef, self.ea, self.eb, self.en, self.ef_ex]:
            w.pack(fill='x', padx=6, pady=1)
        btn(lp, '▶  Calcular', self._run)
        section(lp, 'Puntos de evaluación')
        self.tree = make_table(lp, ['i', 'xᵢ', 'f(xᵢ)', 'peso', 'contribución'], [35,90,90,60,90])
        self.res_lbl = result_label(lp)

    def _n_hint(self): return ''

    def _run(self):
        try:
            expr = parse_math(self.ef.get())
            f = to_f_safe(expr)
            a = fval(self.ea.get())
            b = fval(self.eb.get())
            ok, n = validate_int(self.en.get(), 1)
            if not ok: raise ValueError(n)
            n = self._validate_n(n)
            exact = None
            if self.ef_ex.get().strip():
                try:
                    F = to_f_safe(parse_math(self.ef_ex.get()))
                    exact = float(F(b)) - float(F(a))
                except: pass
            result, points = self._compute(f, a, b, n)
            table_clear(self.tree)
            for i, (xi, fi, wi, ci) in enumerate(points):
                table_insert(self.tree,
                    (i, round(xi,6), round(fi,8), round(wi,4), round(ci,8)),
                    'alt' if i%2 else '')
            msg = f"Integral ≈ {result:.10g}"
            if exact is not None:
                err = abs(result - exact)
                msg += f"\nValor exacto = {exact:.10g}  |  Error = {err:.4e}"
            self.res_lbl.config(text=msg, fg=OK)
            self._plot(expr, a, b, n, result, points, exact)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _validate_n(self, n): return n
    def _compute(self, f, a, b, n): raise NotImplementedError

    def _plot(self, expr, a, b, n, result, points, exact):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        f = to_f_safe(expr)
        xs = np.linspace(a, b, 400)
        try:
            ys = f(xs)
            ax1.plot(xs, ys, 'b-', lw=2, label='f(x)')
        except: pass
        xi_arr = [p[0] for p in points]
        fi_arr = [p[1] for p in points]
        ax1.bar([p[0] for p in points], fi_arr,
                width=(b-a)/max(n,1)*0.8, alpha=0.3, color=ACCENT, label='Rectángulos/Areas')
        ax1.scatter(xi_arr, fi_arr, s=30, color='red', zorder=5)
        ax1.fill_between(xs, 0, np.clip(f(xs), -1e6, 1e6), alpha=0.1, color='blue')
        ax1.set_title(f'Integración — resultado = {result:.6g}', fontsize=9)
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        # Error vs n
        ns = list(range(max(2, self._min_n()), min(102, self._min_n()+50)))
        if exact is not None:
            errs = []
            for ni in ns:
                try:
                    ni = self._validate_n(ni)
                    res, _ = self._compute(f, a, b, ni)
                    errs.append(abs(res - exact))
                except:
                    errs.append(np.nan)
            ax2.semilogy(ns, errs, 'o-', color=ACCENT, lw=1.5, ms=3)
            ax2.axvline(n, color='r', ls='--', label=f'n={n}')
            ax2.set_title('Error vs n', fontsize=9)
            ax2.set_xlabel('n'); ax2.set_ylabel('Error')
            ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Ingrese F(x) antiderivada\npara ver error vs n',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=9)
        self.fig.tight_layout(); self.draw()

    def _min_n(self): return 2


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 – RECTÁNGULO MEDIO
# ══════════════════════════════════════════════════════════════════════════════
class TabRectangulo(NCBase):
    _FLINES = [
        r'$\int_a^b f(x)\,dx \approx h\sum_{i=1}^{n} f\left(\bar{x}_i\right)$',
        r'$\bar{x}_i = \dfrac{x_{i-1}+x_i}{2},\quad h = \dfrac{b-a}{n}$',
        r'$\text{Error} = O(h^2)$',
    ]
    _DESC = (
        "La regla del rectángulo medio evalúa f en el punto medio de cada subintervalo. "
        "Con n subintervalos de ancho h=(b−a)/n, cada rectángulo tiene altura f(x̄ᵢ). "
        "Pese a usar un solo punto por intervalo, el error es O(h²) porque los errores "
        "de subestimación y sobreestimación se cancelan parcialmente."
    )
    def __init__(self, parent):
        super().__init__(parent, '▭ Regla del Rectángulo Medio', self._FLINES, self._DESC)

    def _n_hint(self): return 'n ≥ 1 (cualquier entero positivo)'

    def _compute(self, f, a, b, n):
        h = (b - a) / n
        points = []
        total = 0.0
        for i in range(n):
            xi = a + (i + 0.5) * h
            fi = float(f(xi))
            ci = h * fi
            total += ci
            points.append((xi, fi, 1.0, ci))
        return total, points


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 8 – TRAPECIO
# ══════════════════════════════════════════════════════════════════════════════
class TabTrapecio(NCBase):
    _FLINES = [
        r'$\int_a^b f(x)\,dx \approx \frac{h}{2}\left[f(a)+2\sum_{i=1}^{n-1}f(x_i)+f(b)\right]$',
        r'$h = \dfrac{b-a}{n},\quad E_T = -\dfrac{(b-a)^3}{12n^2}\,f^{\prime\prime}(\xi)$',
    ]
    _DESC = (
        "La regla del trapecio compuesta aproxima la integral conectando puntos consecutivos "
        "con líneas rectas (trapecios). Los puntos extremos tienen peso 1 y los interiores "
        "peso 2. Error global O(h²): se reduce 4× al doblar n. "
        "Caso simple n=1: (b−a)/2·[f(a)+f(b)]."
    )
    def __init__(self, parent):
        super().__init__(parent, '△ Regla del Trapecio', self._FLINES, self._DESC)

    def _n_hint(self): return 'n ≥ 1'

    def _compute(self, f, a, b, n):
        h = (b - a) / n
        xs = np.linspace(a, b, n+1)
        ys = [float(f(x)) for x in xs]
        points = []
        weights = [1.0] + [2.0]*(n-1) + [1.0]
        total = 0.0
        for i in range(n+1):
            ci = (h/2) * weights[i] * ys[i]
            total += ci
            points.append((xs[i], ys[i], weights[i], ci))
        return total, points


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 9 – SIMPSON 1/3
# ══════════════════════════════════════════════════════════════════════════════
class TabSimpson13(NCBase):
    _FLINES = [
        r'$\int_a^b f\,dx \approx \frac{h}{3}\left[f_0+4\sum_{\rm impar}f_i+2\sum_{\rm par}f_i+f_n\right]$',
        r'$h=\dfrac{b-a}{n}\ (n\ \text{par}),\quad E=-\dfrac{(b-a)^5}{180n^4}f^{(4)}(\xi)\;\sim O(h^4)$',
    ]
    _DESC = (
        "Simpson 1/3 ajusta parábolas a cada par de subintervalos (n par obligatorio). "
        "Los pesos alternan 1, 4, 2, 4, 2, …, 4, 1. Error O(h⁴): mucho más preciso que "
        "trapecio. Simple con n=2: h/3·[f(a)+4f(m)+f(b)], h=(b−a)/2."
    )
    def __init__(self, parent):
        super().__init__(parent, '∫ Simpson 1/3', self._FLINES, self._DESC)

    def _n_hint(self): return 'n par (≥ 2)'
    def _min_n(self): return 2

    def _validate_n(self, n):
        if n < 2: raise ValueError("n debe ser ≥ 2")
        if n % 2 != 0:
            n += 1
        return n

    def _compute(self, f, a, b, n):
        n = self._validate_n(n)
        h = (b - a) / n
        xs = np.linspace(a, b, n+1)
        ys = [float(f(x)) for x in xs]
        points = []
        weights = []
        for i in range(n+1):
            if i == 0 or i == n:
                w = 1.0
            elif i % 2 == 1:
                w = 4.0
            else:
                w = 2.0
            weights.append(w)
        total = 0.0
        for i in range(n+1):
            ci = (h/3) * weights[i] * ys[i]
            total += ci
            points.append((xs[i], ys[i], weights[i], ci))
        return total, points


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 10 – SIMPSON 3/8
# ══════════════════════════════════════════════════════════════════════════════
class TabSimpson38(NCBase):
    _FLINES = [
        r'$\int_a^b f\,dx \approx \frac{3h}{8}\left[f_0+3f_1+3f_2+2f_3+\cdots+f_n\right]$',
        r'$h=\dfrac{b-a}{n}\ (n\ \text{múlt. de }3),\quad E\sim O(h^4)$',
    ]
    _DESC = (
        "Simpson 3/8 usa polinomios cúbicos con bloques de 3 subintervalos (n múltiplo de 3). "
        "Los pesos son 1, 3, 3, 2, 3, 3, 2, …, 1. También de orden O(h⁴). "
        "Simple con n=3: 3h/8·[f(a)+3f(x₁)+3f(x₂)+f(b)], h=(b−a)/3."
    )
    def __init__(self, parent):
        super().__init__(parent, '∫ Simpson 3/8', self._FLINES, self._DESC)

    def _n_hint(self): return 'n múltiplo de 3 (≥ 3)'
    def _min_n(self): return 3

    def _validate_n(self, n):
        if n < 3: raise ValueError("n debe ser ≥ 3")
        rem = n % 3
        if rem != 0: n += (3 - rem)
        return n

    def _compute(self, f, a, b, n):
        n = self._validate_n(n)
        h = (b - a) / n
        xs = np.linspace(a, b, n+1)
        ys = [float(f(x)) for x in xs]
        points = []
        total = 0.0
        for i in range(n+1):
            if i == 0 or i == n:
                w = 1.0
            elif i % 3 == 0:
                w = 2.0
            else:
                w = 3.0
            ci = (3*h/8) * w * ys[i]
            total += ci
            points.append((xs[i], ys[i], w, ci))
        return total, points


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 11 – MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
class TabMonteCarlo(TabBase):
    _FLINES = [
        r'$\hat{I} = \dfrac{b-a}{n}\sum_{i=1}^{n} f(x_i),\quad x_i \sim U(a,b)$',
        r'$\sigma_{\hat{I}} = \dfrac{(b-a)\,\sigma_f}{\sqrt{n}},\quad \text{IC} = \hat{I} \pm z_{\alpha/2}\,\sigma_{\hat{I}}$',
        r'$\text{Convergencia: } O(1/\sqrt{n})$',
    ]
    _DESC = (
        "Monte Carlo estima integrales promediando f evaluada en puntos aleatorios uniformes "
        "en [a,b]. La convergencia es O(1/√n): para dividir el error por 10 hay que multiplicar "
        "n por 100. El intervalo de confianza (IC) usa la distribución normal; z vale "
        "1.645 (90%), 1.96 (95%) o 2.576 (99%). Semilla opcional para reproducibilidad."
    )

    def __init__(self, parent):
        super().__init__(parent, '🎲 Monte Carlo')
        lp = self.lp
        formula_card(lp, self._FLINES, self._DESC)

        # ── Mode selector ──────────────────────────────────────────────────────
        section(lp, 'Modo de cálculo')
        self.mc_mode = tk.StringVar(value='simple')
        modes = [('Integral simple', 'simple'), ('Integral doble', 'doble'),
                 ('Área entre curvas (Rechazo)', 'rechazo'), ('Simulaciones especiales', 'simul')]
        for txt, val in modes:
            tk.Radiobutton(lp, text=txt, variable=self.mc_mode, value=val,
                           bg=PANEL, font=(FF,9), command=self._toggle_mc_mode).pack(anchor='w', padx=16, pady=1)

        # ── Frame simple ───────────────────────────────────────────────────────
        self.frm_simple = tk.Frame(lp, bg=PANEL)
        section(self.frm_simple, 'Parámetros — Integral simple')
        self.ef   = LEntry(self.frm_simple, 'f(x) =', 'exp(-x**2)', tooltip='Función a integrar')
        self.ea   = LEntry(self.frm_simple, 'a (límite inf) =', '0')
        self.eb   = LEntry(self.frm_simple, 'b (límite sup) =', '1')
        self.en   = LEntry(self.frm_simple, 'n (muestras) =', '10000', tooltip='Entero positivo')
        self.eseed = LEntry(self.frm_simple, 'Semilla (opt., entero) =', '', tooltip='Dejar vacío para aleatoria')
        tk.Label(self.frm_simple, text='Nivel de confianza:', bg=PANEL, font=(FF,9)).pack(anchor='w', padx=6)
        self.conf_var = tk.StringVar(value='95%')
        ttk.Combobox(self.frm_simple, textvariable=self.conf_var,
                     values=['90%','95%','99%','99.7%'], width=10,
                     state='readonly').pack(anchor='w', padx=6, pady=2)
        for w in [self.ef, self.ea, self.eb, self.en, self.eseed]:
            w.pack(fill='x', padx=6, pady=1)
        btn(self.frm_simple, '▶  Calcular Integral Simple', self._run)
        section(self.frm_simple, 'Convergencia')
        self.tree = make_table(self.frm_simple,
            ['n_muestras', 'Î', 'σ', 'EE', 'IC inferior', 'IC superior'],
            [80, 90, 80, 80, 80, 80])

        # ── Frame doble ────────────────────────────────────────────────────────
        self.frm_doble = tk.Frame(lp, bg=PANEL)
        section(self.frm_doble, 'Parámetros — Integral doble ∫∫ f(x,y) dx dy')
        self.mc2_f  = LEntry(self.frm_doble, 'f(x,y) =', 'exp(-x**2 - y**2)')
        self.mc2_ax = LEntry(self.frm_doble, 'a (x inf) =', '0')
        self.mc2_bx = LEntry(self.frm_doble, 'b (x sup) =', '1')
        self.mc2_ay = LEntry(self.frm_doble, 'a (y inf) =', '0')
        self.mc2_by = LEntry(self.frm_doble, 'b (y sup) =', '1')
        self.mc2_n  = LEntry(self.frm_doble, 'n (muestras) =', '100000')
        self.mc2_seed = LEntry(self.frm_doble, 'Semilla (opt.) =', '')
        tk.Label(self.frm_doble, text='Nivel de confianza:', bg=PANEL, font=(FF,9)).pack(anchor='w', padx=6)
        self.mc2_conf = tk.StringVar(value='95%')
        ttk.Combobox(self.frm_doble, textvariable=self.mc2_conf,
                     values=['90%','95%','99%','99.7%'], width=10, state='readonly').pack(anchor='w', padx=6, pady=2)
        for w in [self.mc2_f, self.mc2_ax, self.mc2_bx, self.mc2_ay, self.mc2_by, self.mc2_n, self.mc2_seed]:
            w.pack(fill='x', padx=6, pady=1)
        btn(self.frm_doble, '▶  Calcular Integral Doble', self._run_doble)
        self.mc2_result = tk.Text(self.frm_doble, height=5, wrap='word', bg='#f0fff0',
                                   font=('Courier New', 9), relief='flat', bd=1, state='disabled')
        self.mc2_result.pack(fill='x', padx=6, pady=4)

        # ── Frame rechazo ──────────────────────────────────────────────────────
        self.frm_rechazo = tk.Frame(lp, bg=PANEL)
        section(self.frm_rechazo, 'Parámetros — Área entre curvas (Método de rechazo)')
        self.mcr_f = LEntry(self.frm_rechazo, 'f(x) (curva superior) =', 'sqrt(x)')
        self.mcr_g = LEntry(self.frm_rechazo, 'g(x) (curva inferior) =', 'x**2')
        self.mcr_a = LEntry(self.frm_rechazo, 'a =', '0')
        self.mcr_b = LEntry(self.frm_rechazo, 'b =', '1')
        self.mcr_n = LEntry(self.frm_rechazo, 'n (muestras) =', '50000')
        self.mcr_seed = LEntry(self.frm_rechazo, 'Semilla (opt.) =', '')
        for w in [self.mcr_f, self.mcr_g, self.mcr_a, self.mcr_b, self.mcr_n, self.mcr_seed]:
            w.pack(fill='x', padx=6, pady=1)
        btn(self.frm_rechazo, '▶  Calcular Área', self._run_rechazo)
        self.mcr_result = tk.Text(self.frm_rechazo, height=5, wrap='word', bg='#f0fff0',
                                   font=('Courier New', 9), relief='flat', bd=1, state='disabled')
        self.mcr_result.pack(fill='x', padx=6, pady=4)

        # ── Frame simul ────────────────────────────────────────────────────────
        self.frm_simul = tk.Frame(lp, bg=PANEL)
        section(self.frm_simul, 'Simulaciones especiales')
        self.simul_mode = tk.StringVar(value='orbital')
        tk.Radiobutton(self.frm_simul, text='Inserción Orbital', variable=self.simul_mode, value='orbital',
                       bg=PANEL, font=(FF,9), command=self._toggle_simul_mode).pack(anchor='w', padx=8)
        tk.Radiobutton(self.frm_simul, text='Black-Scholes / VaR', variable=self.simul_mode, value='bs',
                       bg=PANEL, font=(FF,9), command=self._toggle_simul_mode).pack(anchor='w', padx=8)

        # Orbital frame
        self.frm_orbital = tk.Frame(self.frm_simul, bg=PANEL)
        self.orb_dv     = LEntry(self.frm_orbital, 'Δv nominal (m/s) =', '1000')
        self.orb_dv_std = LEntry(self.frm_orbital, 'σ(Δv) (m/s) =', '10')
        self.orb_dt     = LEntry(self.frm_orbital, 'Δt nominal (s) =', '300')
        self.orb_dt_std = LEntry(self.frm_orbital, 'σ(Δt) (s) =', '2')
        self.orb_n      = LEntry(self.frm_orbital, 'n simulaciones =', '100000')
        self.orb_dv_req = LEntry(self.frm_orbital, 'Δv requerido exacto (m/s) =', '1000')
        for w in [self.orb_dv, self.orb_dv_std, self.orb_dt, self.orb_dt_std, self.orb_n, self.orb_dv_req]:
            w.pack(fill='x', padx=6, pady=1)
        btn(self.frm_orbital, '▶  Simular inserción orbital', self._run_orbital)
        self.orb_result = tk.Text(self.frm_orbital, height=7, wrap='word', bg='#f0fff0',
                                   font=('Courier New', 9), relief='flat', bd=1, state='disabled')
        self.orb_result.pack(fill='x', padx=6, pady=4)
        self.frm_orbital.pack(fill='x', padx=4, pady=2)

        # Black-Scholes frame
        self.frm_bs = tk.Frame(self.frm_simul, bg=PANEL)
        self.bs_s0    = LEntry(self.frm_bs, 'Precio inicial S₀ =', '100')
        self.bs_k     = LEntry(self.frm_bs, 'Strike K =', '105')
        self.bs_r     = LEntry(self.frm_bs, 'Tasa libre de riesgo r =', '0.05')
        self.bs_sigma = LEntry(self.frm_bs, 'Volatilidad σ =', '0.2')
        self.bs_t     = LEntry(self.frm_bs, 'Tiempo T (años) =', '1')
        self.bs_n     = LEntry(self.frm_bs, 'n simulaciones =', '100000')
        self.bs_conf  = LEntry(self.frm_bs, 'Nivel confianza VaR =', '0.99')
        for w in [self.bs_s0, self.bs_k, self.bs_r, self.bs_sigma, self.bs_t, self.bs_n, self.bs_conf]:
            w.pack(fill='x', padx=6, pady=1)
        btn(self.frm_bs, '▶  Calcular opción y VaR', self._run_blackscholes)
        self.bs_result = tk.Text(self.frm_bs, height=7, wrap='word', bg='#f0fff0',
                                  font=('Courier New', 9), relief='flat', bd=1, state='disabled')
        self.bs_result.pack(fill='x', padx=6, pady=4)
        self.frm_bs.pack_forget()

        # ── Common result section ──────────────────────────────────────────────
        self.mc_res_lbl = result_label(lp)
        self.res_lbl = self.mc_res_lbl  # alias for compatibility

        # Show simple mode by default
        self.frm_simple.pack(fill='x', padx=4, pady=2)
        self.frm_doble.pack_forget()
        self.frm_rechazo.pack_forget()
        self.frm_simul.pack_forget()

    def _toggle_mc_mode(self):
        mode = self.mc_mode.get()
        self.frm_simple.pack_forget()
        self.frm_doble.pack_forget()
        self.frm_rechazo.pack_forget()
        self.frm_simul.pack_forget()
        if mode == 'simple':
            self.frm_simple.pack(fill='x', padx=4, pady=2)
        elif mode == 'doble':
            self.frm_doble.pack(fill='x', padx=4, pady=2)
        elif mode == 'rechazo':
            self.frm_rechazo.pack(fill='x', padx=4, pady=2)
        else:
            self.frm_simul.pack(fill='x', padx=4, pady=2)

    def _toggle_simul_mode(self):
        if self.simul_mode.get() == 'orbital':
            self.frm_bs.pack_forget()
            self.frm_orbital.pack(fill='x', padx=4, pady=2)
        else:
            self.frm_orbital.pack_forget()
            self.frm_bs.pack(fill='x', padx=4, pady=2)

    def _run(self):
        mode = self.mc_mode.get()
        if mode == 'simple': self._run_simple()
        elif mode == 'doble': self._run_doble()
        elif mode == 'rechazo': self._run_rechazo()
        elif mode == 'simul':
            if self.simul_mode.get() == 'orbital': self._run_orbital()
            else: self._run_blackscholes()

    def _run_simple(self):
        try:
            expr = parse_math(self.ef.get())
            f = to_f(expr)
            a = fval(self.ea.get())
            b = fval(self.eb.get())
            ok, n = validate_int(self.en.get(), 1)
            if not ok: raise ValueError(n)
            seed_s = self.eseed.get().strip()
            if seed_s:
                np.random.seed(int(seed_s))
            conf_map = {'90%': 1.645, '95%': 1.96, '99%': 2.576, '99.7%': 3.0}
            z = conf_map.get(self.conf_var.get(), 1.96)
            conf_pct = self.conf_var.get()

            xs = np.random.uniform(a, b, n)
            raw = []
            for x in xs:
                try:
                    v = float(f(x))
                    raw.append(v if np.isfinite(v) else np.nan)
                except:
                    raw.append(np.nan)
            fxs = np.array(raw)
            n_finite = int(np.sum(np.isfinite(fxs)))
            if n_finite == 0:
                raise ValueError("f(x) no produjo valores finitos. Verifique la función y el dominio.")
            if n_finite < n:
                fxs = fxs[np.isfinite(fxs)]
                xs  = xs[:len(fxs)]
            n_checks = [10, 50, 100, 500, 1000, 5000, n]
            n_checks = sorted(set([min(ni, n) for ni in n_checks if ni <= n] + [n]))

            table_clear(self.tree)
            results_conv = []
            for ni in n_checks:
                sub = fxs[:ni]
                mean_f = np.mean(sub)
                sigma  = np.std(sub, ddof=1)
                se     = sigma / np.sqrt(ni)
                est    = (b - a) * mean_f
                lo     = est - z * (b-a) * se
                hi     = est + z * (b-a) * se
                table_insert(self.tree,
                    (ni, round(est,8), round(sigma,6), round(se,6),
                     round(lo,8), round(hi,8)), 'alt' if len(results_conv)%2 else '')
                results_conv.append((ni, est, lo, hi))

            final_est = (b - a) * np.mean(fxs)
            sigma_f = np.std(fxs, ddof=1)
            se_f = sigma_f / np.sqrt(n)
            lo_f = final_est - z * (b-a) * se_f
            hi_f = final_est + z * (b-a) * se_f
            self.res_lbl.config(
                text=(f"Î = {final_est:.10g}\n"
                      f"IC {conf_pct}: [{lo_f:.8g}, {hi_f:.8g}]\n"
                      f"σ={sigma_f:.6g}  EE={se_f:.6g}  Amplitud IC={hi_f-lo_f:.6g}"),
                fg=OK)
            self._plot(xs, fxs, a, b, results_conv, z, conf_pct)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _plot(self, xs, fxs, a, b, results_conv, z, conf_pct):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        # Scatter
        n_show = min(len(xs), 500)
        idx = np.random.choice(len(xs), n_show, replace=False)
        inside = fxs[idx] >= 0
        ax1.scatter(xs[idx][inside], fxs[idx][inside], s=4, color='blue', alpha=0.5, label='f(xᵢ)>0')
        ax1.scatter(xs[idx][~inside], fxs[idx][~inside], s=4, color='red', alpha=0.5)
        ax1.axhline(0, color='k', lw=0.7)
        ax1.set_title(f'Muestreo Monte Carlo (n={len(xs)})', fontsize=9)
        ax1.set_xlabel('x'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        # Convergence
        ns_plot = [r[0] for r in results_conv]
        ests    = [r[1] for r in results_conv]
        los     = [r[2] for r in results_conv]
        his     = [r[3] for r in results_conv]
        ax2.plot(ns_plot, ests, 'b-o', lw=1.5, ms=4, label='Î')
        ax2.fill_between(ns_plot, los, his, alpha=0.2, color='blue', label=f'IC {conf_pct}')
        ax2.set_title('Convergencia del estimador', fontsize=9)
        ax2.set_xlabel('n muestras'); ax2.set_ylabel('Î')
        ax2.set_xscale('log'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()

    def _run_doble(self):
        try:
            x_sym, y_sym = sp.Symbol('x'), sp.Symbol('y')
            expr = parse_math(self.mc2_f.get(), extra={'x': x_sym, 'y': y_sym})
            f = to_f2(expr, 'x', 'y')
            ax = fval(self.mc2_ax.get()); bx = fval(self.mc2_bx.get())
            ay = fval(self.mc2_ay.get()); by = fval(self.mc2_by.get())
            ok, n = validate_int(self.mc2_n.get(), 1)
            if not ok: raise ValueError(n)
            seed_s = self.mc2_seed.get().strip()
            if seed_s: np.random.seed(int(seed_s))
            conf_map = {'90%':1.645,'95%':1.96,'99%':2.576,'99.7%':3.0}
            z = conf_map.get(self.mc2_conf.get(), 1.96)

            xs = np.random.uniform(ax, bx, n)
            ys = np.random.uniform(ay, by, n)
            fxys = np.array([float(f(xi, yi)) for xi, yi in zip(xs, ys)])
            finite = np.isfinite(fxys)
            fxys_f = fxys[finite]
            vol = (bx-ax)*(by-ay)
            est = vol * np.mean(fxys_f)
            sigma = np.std(fxys_f, ddof=1)
            se = sigma / np.sqrt(len(fxys_f))
            lo = est - z*(vol)*se; hi = est + z*(vol)*se

            text = (f"∫∫ f(x,y) dx dy ≈ {est:.10g}\n"
                    f"IC {self.mc2_conf.get()}: [{lo:.8g}, {hi:.8g}]\n"
                    f"σ = {sigma:.6g}  EE = {se:.6g}  n válidos = {np.sum(finite)}")
            self.mc2_result.config(state='normal')
            self.mc2_result.delete('1.0','end')
            self.mc2_result.insert('1.0', text)
            self.mc2_result.config(state='disabled')
            self.res_lbl.config(text=f"Integral doble ≈ {est:.8g}", fg=OK)

            # Plot scatter
            self.fig.clear()
            ax1 = self.fig.add_subplot(111)
            n_show = min(3000, len(xs))
            idx = np.random.choice(len(xs), n_show, replace=False)
            sc = ax1.scatter(xs[idx], ys[idx], c=fxys[idx], cmap='viridis', s=2, alpha=0.6)
            self.fig.colorbar(sc, ax=ax1, label='f(x,y)')
            ax1.set_xlabel('x'); ax1.set_ylabel('y')
            ax1.set_title(f'Monte Carlo Doble — Î = {est:.6g}', fontsize=9)
            ax1.grid(True, alpha=0.3)
            self.fig.tight_layout(); self.draw()
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _run_rechazo(self):
        try:
            expr_f = parse_math(self.mcr_f.get())
            expr_g = parse_math(self.mcr_g.get())
            f_func = to_f_safe(expr_f)
            g_func = to_f_safe(expr_g)
            a = fval(self.mcr_a.get()); b = fval(self.mcr_b.get())
            ok, n = validate_int(self.mcr_n.get(), 1)
            if not ok: raise ValueError(n)
            seed_s = self.mcr_seed.get().strip()
            if seed_s: np.random.seed(int(seed_s))

            # Find y bounds
            xtest = np.linspace(a, b, 500)
            ftest = f_func(xtest); gtest = g_func(xtest)
            ymin = float(np.min(gtest)); ymax = float(np.max(ftest))

            xs_r = np.random.uniform(a, b, n)
            ys_r = np.random.uniform(ymin, ymax, n)
            fv = f_func(xs_r); gv = g_func(xs_r)
            inside = (ys_r <= fv) & (ys_r >= gv)

            prop = np.sum(inside) / n
            area_box = (b-a) * (ymax-ymin)
            area_est = prop * area_box
            se = np.sqrt(prop*(1-prop)/n) * area_box

            # Also compute analytically via sympy
            x_sym = sp.Symbol('x')
            try:
                area_exact = float(sp.integrate(expr_f - expr_g, (x_sym, a, b)).evalf())
                error_str = f"\nError vs analítico: {abs(area_est-area_exact):.4e}"
            except:
                area_exact = None
                error_str = ""

            text = (f"Área entre curvas ≈ {area_est:.8g}\n"
                    f"Puntos aceptados: {np.sum(inside)}/{n} ({prop*100:.2f}%)\n"
                    f"Error estándar: {se:.6g}" + error_str)
            self.mcr_result.config(state='normal')
            self.mcr_result.delete('1.0','end')
            self.mcr_result.insert('1.0', text)
            self.mcr_result.config(state='disabled')
            self.res_lbl.config(text=f"Área entre curvas ≈ {area_est:.6g}", fg=OK)

            # Plot
            self.fig.clear()
            ax1 = self.fig.add_subplot(111)
            xp = np.linspace(a, b, 400)
            ax1.plot(xp, f_func(xp), 'b-', lw=2, label='f(x) superior')
            ax1.plot(xp, g_func(xp), 'g-', lw=2, label='g(x) inferior')
            ax1.fill_between(xp, g_func(xp), f_func(xp), alpha=0.2, color='blue', label=f'Área≈{area_est:.4g}')
            n_show = min(3000, n)
            idx = np.random.choice(n, n_show, replace=False)
            mask_in = inside[idx]; mask_out = ~inside[idx]
            ax1.scatter(xs_r[idx][mask_in], ys_r[idx][mask_in], s=2, color='blue', alpha=0.3, label='Aceptado')
            ax1.scatter(xs_r[idx][mask_out], ys_r[idx][mask_out], s=2, color='red', alpha=0.3, label='Rechazado')
            ax1.set_title('Método de rechazo — área entre curvas', fontsize=9)
            ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)
            self.fig.tight_layout(); self.draw()
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _run_orbital(self):
        try:
            dv_nom = fval(self.orb_dv.get())
            dv_std = fval(self.orb_dv_std.get())
            dt_nom = fval(self.orb_dt.get())
            dt_std = fval(self.orb_dt_std.get())
            ok, n = validate_int(self.orb_n.get(), 100)
            if not ok: raise ValueError(n)
            dv_req = fval(self.orb_dv_req.get())

            np.random.seed(42)
            dv_samples = np.random.normal(dv_nom, dv_std, n)
            dt_samples = np.random.normal(dt_nom, dt_std, n)
            # Effective delta-v = dv * (dt/dt_nom), simplified model
            dv_eff = dv_samples * (dt_samples / dt_nom)
            successes = dv_eff >= dv_req
            p_success = np.mean(successes)

            # Margin for 99% success
            try:
                from scipy import stats as scipy_stats
                margin = scipy_stats.norm.ppf(0.99) * dv_std - (dv_nom - dv_req)
            except ImportError:
                # fallback: z=2.326 for 99%
                margin = 2.326 * dv_std - (dv_nom - dv_req)
            dv_99 = dv_req + max(0, margin)

            text = (f"Δv nominal: {dv_nom} m/s  σ(Δv): {dv_std} m/s\n"
                    f"Δt nominal: {dt_nom} s  σ(Δt): {dt_std} s\n"
                    f"Δv requerido: {dv_req} m/s\n"
                    f"P(éxito) = {p_success*100:.2f}%  ({np.sum(successes)}/{n} casos)\n"
                    f"Δv para 99% éxito ≈ {dv_99:.2f} m/s\n"
                    f"Margen adicional ≈ {max(0, dv_99-dv_nom):.2f} m/s")
            self.orb_result.config(state='normal')
            self.orb_result.delete('1.0','end')
            self.orb_result.insert('1.0', text)
            self.orb_result.config(state='disabled')
            self.res_lbl.config(text=f"P(éxito) = {p_success*100:.2f}%", fg=OK)

            self.fig.clear()
            ax1 = self.fig.add_subplot(211)
            ax2 = self.fig.add_subplot(212)
            ax1.hist(dv_eff, bins=60, color=ACCENT, alpha=0.7, edgecolor='white', lw=0.3)
            ax1.axvline(dv_req, color='red', lw=2, label=f'Δv req={dv_req}')
            ax1.axvline(np.mean(dv_eff), color='green', ls='--', lw=1.5, label=f'Media={np.mean(dv_eff):.1f}')
            ax1.set_title(f'Distribución Δv efectivo — P(éxito)={p_success*100:.1f}%', fontsize=9)
            ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
            ns = np.logspace(1, np.log10(n), 50, dtype=int)
            ps = [np.mean(dv_eff[:ni] >= dv_req) for ni in ns]
            ax2.semilogx(ns, [p*100 for p in ps], 'b-', lw=1.5)
            ax2.axhline(p_success*100, color='r', ls='--', lw=1)
            ax2.set_xlabel('n muestras'); ax2.set_ylabel('P(éxito) %')
            ax2.set_title('Convergencia de P(éxito)', fontsize=9)
            ax2.grid(True, alpha=0.3)
            self.fig.tight_layout(); self.draw()
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _run_blackscholes(self):
        try:
            S0 = fval(self.bs_s0.get())
            K  = fval(self.bs_k.get())
            r  = fval(self.bs_r.get())
            sigma = fval(self.bs_sigma.get())
            T  = fval(self.bs_t.get())
            ok, n = validate_int(self.bs_n.get(), 1000)
            if not ok: raise ValueError(n)
            conf = fval(self.bs_conf.get())

            np.random.seed(42)
            Z = np.random.standard_normal(n)
            ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
            payoffs = np.maximum(ST - K, 0)
            price_mc = np.exp(-r*T) * np.mean(payoffs)
            price_se = np.exp(-r*T) * np.std(payoffs, ddof=1) / np.sqrt(n)

            # Black-Scholes analytical formula
            try:
                from scipy.stats import norm as spnorm
                d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                price_bs = S0*spnorm.cdf(d1) - K*np.exp(-r*T)*spnorm.cdf(d2)
            except ImportError:
                # Manual normal CDF approximation
                def _norm_cdf(x):
                    return 0.5 * (1 + np.sign(x) * (1 - np.exp(-0.7 * x**2))**0.5)
                d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                price_bs = S0*_norm_cdf(d1) - K*np.exp(-r*T)*_norm_cdf(d2)

            # Portfolio: 1 stock + 1 call option
            port_vals = ST + payoffs * np.exp(-r*T)
            port_loss = -(port_vals - (S0 + price_mc))
            var_pct = conf
            var_val = np.percentile(port_loss, var_pct*100)

            text = (f"Precio call (MC):       {price_mc:.6f}  ±{1.96*price_se:.6f}\n"
                    f"Precio call (B-S exact): {price_bs:.6f}\n"
                    f"Error MC vs B-S:         {abs(price_mc-price_bs):.6f}\n"
                    f"VaR {conf*100:.0f}% a 1 día:     {var_val:.4f}\n"
                    f"(pérdida máxima con {conf*100:.0f}% de confianza)")
            self.bs_result.config(state='normal')
            self.bs_result.delete('1.0','end')
            self.bs_result.insert('1.0', text)
            self.bs_result.config(state='disabled')
            self.res_lbl.config(text=f"Call MC={price_mc:.4f}  B-S={price_bs:.4f}  VaR={var_val:.4f}", fg=OK)

            self.fig.clear()
            ax1 = self.fig.add_subplot(211)
            ax2 = self.fig.add_subplot(212)
            ax1.hist(ST, bins=60, color='#2a6ebb', alpha=0.7, edgecolor='white', lw=0.3, density=True)
            ax1.axvline(K, color='red', lw=2, label=f'K={K}')
            ax1.axvline(S0, color='green', ls='--', lw=1.5, label=f'S₀={S0}')
            ax1.set_title(f'Distribución S(T) — Precio Call MC={price_mc:.4f}', fontsize=9)
            ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
            losses_sorted = np.sort(port_loss)
            ax2.plot(np.linspace(0,1,n), losses_sorted, 'b-', lw=1)
            ax2.axhline(var_val, color='red', lw=1.5, label=f'VaR {conf*100:.0f}%={var_val:.3f}')
            ax2.axvline(conf, color='orange', ls='--', lw=1)
            ax2.set_xlabel('Percentil'); ax2.set_ylabel('Pérdida')
            ax2.set_title('Distribución de pérdidas — VaR', fontsize=9)
            ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
            self.fig.tight_layout(); self.draw()
        except Exception as ex:
            messagebox.showerror('Error', str(ex))


# ══════════════════════════════════════════════════════════════════════════════
#  ODE BASE
# ══════════════════════════════════════════════════════════════════════════════
class ODEBase(TabBase):
    def __init__(self, parent, title, flines, desc):
        super().__init__(parent, title)
        lp = self.lp
        formula_card(lp, flines, desc)
        section(lp, 'Parámetros')
        self.ef   = LEntry(lp, "f(x,y) = dy/dx =", 'x + y', tooltip='Ej: -y + x, y*sin(x)')
        self.ey0  = LEntry(lp, 'y₀ (cond. inicial) =', '1.0')
        self.ex0  = LEntry(lp, 'x₀ (inicio) =', '0.0')
        self.exf  = LEntry(lp, 'xf (fin) =', '1.0')
        self.eh   = LEntry(lp, 'h (paso) =', '0.1', tooltip='h > 0')
        self.eyex = LEntry(lp, 'y_exact(x) (opt.) =', '-x-1+2*exp(x)', tooltip='Solución exacta para comparar')
        for w in [self.ef, self.ey0, self.ex0, self.exf, self.eh, self.eyex]:
            w.pack(fill='x', padx=6, pady=1)
        btn(lp, '▶  Calcular', self._run)
        section(lp, 'Iteraciones')
        cols = self._cols()
        self.tree = make_table(lp, cols, [40] + [90]*(len(cols)-1))
        self.res_lbl = result_label(lp)

    def _cols(self): return ['n', 'x', 'y_aprox', 'y_exact', 'Error']

    def _run(self):
        try:
            expr2 = parse_math(self.ef.get(), extra={'x': sp.Symbol('x'), 'y': sp.Symbol('y')})
            f = to_f2(expr2, 'x', 'y')
            y0 = fval(self.ey0.get())
            x0 = fval(self.ex0.get())
            xf = fval(self.exf.get())
            h  = fval(self.eh.get())
            if h <= 0: raise ValueError("h debe ser > 0")
            if x0 >= xf: raise ValueError("x₀ < xf requerido")
            has_ex = bool(self.eyex.get().strip())
            y_ex_f = None
            if has_ex:
                try:
                    y_ex_f = to_f(parse_math(self.eyex.get()))
                except: pass
            xs = np.arange(x0, xf + h*0.5, h)
            ys = self._solve(f, y0, xs, h)
            table_clear(self.tree)
            rows_plot = []
            for i, (xi, yi) in enumerate(zip(xs, ys)):
                ye = float(y_ex_f(xi)) if y_ex_f else None
                err = abs(yi - ye) if ye is not None else None
                row = self._make_row(i, xi, yi, ye, err)
                table_insert(self.tree, row, 'alt' if i%2 else '')
                rows_plot.append((xi, yi, ye))
            self.res_lbl.config(
                text=f"y({xs[-1]:.4g}) ≈ {ys[-1]:.10g}" +
                     (f"  |  y_exact = {float(y_ex_f(xs[-1])):.10g}" if y_ex_f else ''),
                fg=OK)
            self._plot(xs, ys, y_ex_f, rows_plot)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _make_row(self, i, xi, yi, ye, err):
        row = [i, round(xi, 6), round(yi, 8)]
        if ye is not None:
            row += [round(ye, 8), f"{err:.4e}"]
        else:
            row += ['—', '—']
        return row

    def _solve(self, f, y0, xs, h): raise NotImplementedError

    def _plot(self, xs, ys, y_ex_f, rows_plot):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        ax1.plot(xs, ys, 'b-o', lw=1.5, ms=4, label='Aprox.')
        if y_ex_f:
            xs_fine = np.linspace(xs[0], xs[-1], 400)
            try:
                ax1.plot(xs_fine, y_ex_f(xs_fine), 'g--', lw=2, label='Exacta')
            except: pass
            errs = [abs(yi - float(y_ex_f(xi))) for xi, yi in zip(xs, ys)]
            ax2.semilogy(xs, errs, 'r-o', lw=1.5, ms=4)
            ax2.set_title('Error absoluto', fontsize=9)
            ax2.set_xlabel('x'); ax2.set_ylabel('|error|')
            ax2.grid(True, alpha=0.3)
        ax1.set_title('Solución ODE', fontsize=9)
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 12 – EULER
# ══════════════════════════════════════════════════════════════════════════════
class TabEuler(ODEBase):
    _FLINES = [
        r'$\dfrac{dy}{dx} = f(x,y),\quad y(x_0)=y_0$',
        r'$y_{n+1} = y_n + h\,f(x_n,\,y_n)$',
        r'$\text{Error global: } O(h)$',
    ]
    _DESC = (
        "Euler avanza proyectando la pendiente f(xₙ,yₙ) desde el punto actual. "
        "Es el método más simple de paso único para EDO de valor inicial. "
        "El error de truncamiento local es O(h²) y el global O(h): poco preciso para h grande. "
        "Se necesita h pequeño para obtener buena exactitud."
    )
    def __init__(self, parent):
        super().__init__(parent, '→ Euler', self._FLINES, self._DESC)

    def _solve(self, f, y0, xs, h):
        ys = [y0]
        for i in range(len(xs)-1):
            yn = ys[-1] + h * float(f(xs[i], ys[-1]))
            ys.append(yn)
        return ys


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 13 – EULER MEJORADO (HEUN)
# ══════════════════════════════════════════════════════════════════════════════
class TabHeun(ODEBase):
    _FLINES = [
        r'$y^* = y_n + h\,f(x_n,y_n)\quad\leftarrow$ predicción',
        r'$y_{n+1} = y_n + \dfrac{h}{2}\left[f(x_n,y_n)+f(x_n+h,\,y^*)\right]$',
        r'$\text{Error global: } O(h^2)$',
    ]
    _DESC = (
        "Heun (Euler mejorado / RK2) usa un paso predictor-corrector: primero estima y* "
        "con Euler, luego promedia las pendientes en xₙ e xₙ₊₁ para corregir. "
        "Error global O(h²): mucho más preciso que Euler simple con el mismo h. "
        "Equivalente a la regla del trapecio para EDO."
    )
    def __init__(self, parent):
        super().__init__(parent, '⇒ Euler Mejorado (Heun)', self._FLINES, self._DESC)

    def _solve(self, f, y0, xs, h):
        ys = [y0]
        for i in range(len(xs)-1):
            k1 = float(f(xs[i], ys[-1]))
            y_pred = ys[-1] + h * k1
            k2 = float(f(xs[i]+h, y_pred))
            ys.append(ys[-1] + h/2*(k1 + k2))
        return ys


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 14 – RUNGE-KUTTA 4
# ══════════════════════════════════════════════════════════════════════════════
class TabRK4(TabBase):
    _FLINES = [
        r'$k_1=f(x_n,y_n),\ k_2=f\left(x_n+\frac{h}{2},y_n+\frac{h}{2}k_1\right)$',
        r'$k_3=f\left(x_n+\frac{h}{2},y_n+\frac{h}{2}k_2\right),\ k_4=f(x_n+h,\,y_n+hk_3)$',
        r'$y_{n+1}=y_n+\dfrac{h}{6}(k_1+2k_2+2k_3+k_4),\quad O(h^4)$',
    ]
    _DESC = (
        "RK4 es el estándar de facto para integración numérica de EDO. Evalúa f en 4 puntos "
        "por paso: inicio, dos puntos medios y el final, ponderados 1,2,2,1. "
        "Error global O(h⁴): enormemente más preciso que Euler (O(h)) y Heun (O(h²)). "
        "Muy recomendado para problemas de ingeniería donde se necesite precisión."
    )

    def __init__(self, parent):
        super().__init__(parent, '🚀 Runge-Kutta 4')
        lp = self.lp
        formula_card(lp, self._FLINES, self._DESC)
        section(lp, 'Parámetros')
        self.ef   = LEntry(lp, "f(x,y) = dy/dx =", 'x + y')
        self.ey0  = LEntry(lp, 'y₀ =', '1.0')
        self.ex0  = LEntry(lp, 'x₀ =', '0.0')
        self.exf  = LEntry(lp, 'xf =', '1.0')
        self.eh   = LEntry(lp, 'h =', '0.1')
        self.eyex = LEntry(lp, 'y_exact(x) (opt.) =', '-x-1+2*exp(x)')
        for w in [self.ef, self.ey0, self.ex0, self.exf, self.eh, self.eyex]:
            w.pack(fill='x', padx=6, pady=1)
        btn(lp, '▶  Calcular', self._run)
        section(lp, 'Iteraciones con pendientes k₁..k₄')
        self.tree = make_table(lp,
            ['n','x','k₁','k₂','k₃','k₄','yₙ₊₁','y_exact','Error'],
            [30,70,75,75,75,75,80,80,70])
        self.res_lbl = result_label(lp)

    def _run(self):
        try:
            expr2 = parse_math(self.ef.get(), extra={'x': sp.Symbol('x'), 'y': sp.Symbol('y')})
            f = to_f2(expr2, 'x', 'y')
            y0 = fval(self.ey0.get())
            x0 = fval(self.ex0.get())
            xf = fval(self.exf.get())
            h  = fval(self.eh.get())
            if h <= 0: raise ValueError("h > 0 requerido")
            has_ex = bool(self.eyex.get().strip())
            y_ex_f = None
            if has_ex:
                try: y_ex_f = to_f(parse_math(self.eyex.get()))
                except: pass
            xs = np.arange(x0, xf + h*0.5, h)
            ys = [y0]
            k_data = []
            for i in range(len(xs)-1):
                xi, yi = xs[i], ys[-1]
                k1 = float(f(xi, yi))
                k2 = float(f(xi+h/2, yi+h/2*k1))
                k3 = float(f(xi+h/2, yi+h/2*k2))
                k4 = float(f(xi+h, yi+h*k3))
                yn1 = yi + h/6*(k1 + 2*k2 + 2*k3 + k4)
                ys.append(yn1)
                k_data.append((xi, k1, k2, k3, k4, yn1))
            table_clear(self.tree)
            for i, (xi, k1, k2, k3, k4, yn1) in enumerate(k_data):
                ye = float(y_ex_f(xs[i+1])) if y_ex_f else None
                err = abs(yn1 - ye) if ye is not None else '—'
                table_insert(self.tree,
                    (i+1, round(xs[i+1],5), round(k1,7), round(k2,7),
                     round(k3,7), round(k4,7), round(yn1,8),
                     round(ye,8) if ye else '—',
                     f"{err:.4e}" if isinstance(err, float) else err),
                    'alt' if i%2 else '')
            self.res_lbl.config(
                text=f"y({xs[-1]:.4g}) ≈ {ys[-1]:.10g}" +
                     (f"  |  exacta = {float(y_ex_f(xs[-1])):.10g}" if y_ex_f else ''),
                fg=OK)
            self._plot(xs, ys, y_ex_f)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _plot(self, xs, ys, y_ex_f):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        ax1.plot(xs, ys, 'b-o', lw=1.5, ms=4, label='RK4')
        if y_ex_f:
            xs_fine = np.linspace(xs[0], xs[-1], 400)
            try:
                ax1.plot(xs_fine, y_ex_f(xs_fine), 'g--', lw=2, label='Exacta')
            except: pass
            errs = [abs(yi - float(y_ex_f(xi))) for xi, yi in zip(xs, ys)]
            ax2.semilogy(xs, errs, 'r-o', lw=1.5, ms=4, label='Error RK4')
            ax2.set_title('Error absoluto RK4', fontsize=9)
            ax2.set_xlabel('x'); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)
        ax1.set_title('Runge-Kutta 4', fontsize=9)
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 15 – COMPARADOR RK (Euler vs Heun vs RK4)
# ══════════════════════════════════════════════════════════════════════════════
class TabComparador(TabBase):
    _FLINES = [
        r'$\text{Euler: } y_{n+1}=y_n+h\,f_n \quad O(h)$',
        r'$\text{Heun: } y_{n+1}=y_n+\frac{h}{2}(f_n+f_{n+1}^*) \quad O(h^2)$',
        r'$\text{RK4: } y_{n+1}=y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4) \quad O(h^4)$',
    ]
    _DESC = (
        "Compara los tres métodos para la misma EDO, condiciones iniciales y paso h. "
        "Euler es el más simple pero menos preciso; Heun (RK2) es cuadrático; "
        "RK4 es el estándar con error de orden 4. "
        "Con solución exacta, se grafican los errores absolutos de cada método para comparar."
    )
    def __init__(self, parent):
        super().__init__(parent, '⚖ Comparador: Euler / Heun / RK4')
        lp = self.lp
        formula_card(lp, self._FLINES, self._DESC)
        section(lp, 'Parámetros')
        self.ef   = LEntry(lp, "f(x,y) =", 'x + y')
        self.ey0  = LEntry(lp, 'y₀ =', '1.0')
        self.ex0  = LEntry(lp, 'x₀ =', '0.0')
        self.exf  = LEntry(lp, 'xf =', '1.0')
        self.eh   = LEntry(lp, 'h =', '0.1')
        self.eyex = LEntry(lp, 'y_exact(x) (opt.) =', '-x-1+2*exp(x)')
        for w in [self.ef, self.ey0, self.ex0, self.exf, self.eh, self.eyex]:
            w.pack(fill='x', padx=6, pady=1)
        btn(lp, '▶  Calcular', self._run)
        section(lp, 'Comparación')
        self.tree = make_table(lp,
            ['n','x','Euler','Heun','RK4','Exacta','Err Euler','Err Heun','Err RK4'],
            [30,65,80,80,80,80,75,75,75])
        self.res_lbl = result_label(lp)

    def _run(self):
        try:
            expr2 = parse_math(self.ef.get(), extra={'x': sp.Symbol('x'), 'y': sp.Symbol('y')})
            f = to_f2(expr2, 'x', 'y')
            y0 = fval(self.ey0.get())
            x0, xf = fval(self.ex0.get()), fval(self.exf.get())
            h = fval(self.eh.get())
            if h <= 0: raise ValueError("h > 0")
            has_ex = bool(self.eyex.get().strip())
            yef = None
            if has_ex:
                try: yef = to_f(parse_math(self.eyex.get()))
                except: pass
            xs = np.arange(x0, xf + h*0.5, h)
            ye_arr = [y0]; yh_arr = [y0]; yrk_arr = [y0]
            for i in range(len(xs)-1):
                xi = xs[i]
                # Euler
                ye_arr.append(ye_arr[-1] + h * float(f(xi, ye_arr[-1])))
                # Heun
                k1h = float(f(xi, yh_arr[-1]))
                yp = yh_arr[-1] + h*k1h
                yh_arr.append(yh_arr[-1] + h/2*(k1h + float(f(xi+h, yp))))
                # RK4
                yi = yrk_arr[-1]
                k1 = float(f(xi, yi))
                k2 = float(f(xi+h/2, yi+h/2*k1))
                k3 = float(f(xi+h/2, yi+h/2*k2))
                k4 = float(f(xi+h, yi+h*k3))
                yrk_arr.append(yi + h/6*(k1+2*k2+2*k3+k4))
            table_clear(self.tree)
            for i in range(len(xs)):
                ye = float(yef(xs[i])) if yef else None
                def ferr(y): return f"{abs(y-(ye if ye else y)):.4e}" if ye else '—'
                table_insert(self.tree,
                    (i, round(xs[i],5), round(ye_arr[i],7), round(yh_arr[i],7),
                     round(yrk_arr[i],7), round(ye,8) if ye else '—',
                     ferr(ye_arr[i]), ferr(yh_arr[i]), ferr(yrk_arr[i])),
                    'alt' if i%2 else '')
            self.res_lbl.config(text=f"Resultados finales en x={xs[-1]:.4g}: "
                f"Euler={ye_arr[-1]:.6g} | Heun={yh_arr[-1]:.6g} | RK4={yrk_arr[-1]:.6g}", fg=OK)
            self._plot(xs, ye_arr, yh_arr, yrk_arr, yef)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _plot(self, xs, ye, yh, yrk, yef):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        ax1.plot(xs, ye, 'r-o', lw=1.2, ms=3, label='Euler')
        ax1.plot(xs, yh, 'g-s', lw=1.2, ms=3, label='Heun')
        ax1.plot(xs, yrk, 'b-^', lw=1.2, ms=3, label='RK4')
        if yef:
            xs_f = np.linspace(xs[0], xs[-1], 400)
            try: ax1.plot(xs_f, yef(xs_f), 'k--', lw=2, label='Exacta')
            except: pass
            ae = [abs(ye[i]-float(yef(xs[i]))) for i in range(len(xs))]
            ah = [abs(yh[i]-float(yef(xs[i]))) for i in range(len(xs))]
            ar = [abs(yrk[i]-float(yef(xs[i]))) for i in range(len(xs))]
            ax2.semilogy(xs, ae, 'r-o', lw=1.2, ms=3, label='Error Euler')
            ax2.semilogy(xs, ah, 'g-s', lw=1.2, ms=3, label='Error Heun')
            ax2.semilogy(xs, ar, 'b-^', lw=1.2, ms=3, label='Error RK4')
            ax2.set_title('Errores comparados', fontsize=9)
            ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
        ax1.set_title('Comparación métodos ODE', fontsize=9)
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 16 – CALCULADORA INTELIGENTE
# ══════════════════════════════════════════════════════════════════════════════
class TabCalculadora(TabBase):
    def __init__(self, parent):
        super().__init__(parent, '🧮 Calculadora Inteligente — Derivadas, Integrales y Límites')
        lp = self.lp

        theory_box(lp,
            "Calculadora simbólica usando SymPy para resultados analíticos exactos.\n"
            "DERIVADAS: cualquier orden, con evaluación en un punto opcional.\n"
            "INTEGRALES: indefinida (f dx), definida (∫ₐᵇ f dx), impropia (a/b = oo/∞).\n"
            "LÍMITES: bilateral, unilateral derecho (+) o izquierdo (−), en ±∞.\n\n"
            "Notación amigable: 2x→2*x, x^2→x², ln→log, sen→sin, √→sqrt, e→exp(1)."
        )

        # ── DERIVADAS ──────────────────────────────────────────────────────────
        section(lp, '─── DERIVADAS ───')
        self.dv_expr  = LEntry(lp, 'f(x) =', 'x**3 * sin(x)', tooltip='Función a derivar')
        self.dv_var   = LEntry(lp, 'Variable =', 'x', tooltip='Variable de diferenciación')
        self.dv_order = LEntry(lp, 'Orden (n) =', '1', tooltip='1 = primera derivada, 2 = segunda, …')
        self.dv_pt    = LEntry(lp, 'Evaluar en x = (opt.) =', '', tooltip='Dejar vacío para resultado simbólico')
        for w in [self.dv_expr, self.dv_var, self.dv_order, self.dv_pt]:
            w.pack(fill='x', padx=6, pady=1)
        btn(lp, "f'  Calcular Derivada", self._calc_deriv)
        self.dv_result = tk.Text(lp, height=6, wrap='word', bg='#fff8f0',
                                  font=('Courier New', 9), relief='flat', bd=1)
        self.dv_result.pack(fill='x', padx=6, pady=4)

        section(lp, '─── INTEGRALES ───')
        self.if_expr = LEntry(lp, 'f(x) =', 'sin(x)*exp(-x)', tooltip='Función a integrar')
        self.if_var  = LEntry(lp, 'Variable =', 'x', tooltip='Variable de integración')
        self.if_a    = LEntry(lp, 'Límite inf a =', '0', tooltip='Dejar vacío para indefinida')
        self.if_b    = LEntry(lp, 'Límite sup b =', 'oo', tooltip='Usar "oo" para infinito')
        self.if_indef = tk.BooleanVar(value=False)
        tk.Checkbutton(lp, text='Integral indefinida (sin límites)',
                       variable=self.if_indef, bg=PANEL, font=(FF,9)).pack(anchor='w', padx=6)
        for w in [self.if_expr, self.if_var, self.if_a, self.if_b]:
            w.pack(fill='x', padx=6, pady=1)
        btn(lp, '∫  Calcular Integral', self._calc_integral)

        self.int_result = tk.Text(lp, height=5, wrap='word', bg='#f0fff0',
                                   font=('Courier New', 9), relief='flat', bd=1)
        self.int_result.pack(fill='x', padx=6, pady=4)

        section(lp, '─── LÍMITES ───')
        self.lm_expr = LEntry(lp, 'f(x) =', 'sin(x)/x', tooltip='Función para el límite')
        self.lm_var  = LEntry(lp, 'Variable =', 'x')
        self.lm_pt   = LEntry(lp, 'x → =', '0', tooltip='Punto: número, oo, -oo')
        tk.Label(lp, text='Dirección:', bg=PANEL, font=(FF,9)).pack(anchor='w', padx=6)
        self.lm_dir = tk.StringVar(value='bilateral')
        for txt, val in [('Bilateral (ambos lados)','bilateral'),
                         ('Derecho (+)','right'),
                         ('Izquierdo (−)','left')]:
            tk.Radiobutton(lp, text=txt, variable=self.lm_dir, value=val,
                           bg=PANEL, font=(FF,8)).pack(anchor='w', padx=20)
        for w in [self.lm_expr, self.lm_var, self.lm_pt]:
            w.pack(fill='x', padx=6, pady=1)
        btn(lp, 'lim  Calcular Límite', self._calc_limit)

        self.lm_result = tk.Text(lp, height=4, wrap='word', bg='#f0fff0',
                                  font=('Courier New', 9), relief='flat', bd=1)
        self.lm_result.pack(fill='x', padx=6, pady=4)

    def _show_text(self, widget, text):
        widget.config(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)
        widget.config(state='disabled')

    def _calc_deriv(self):
        try:
            expr    = parse_math(self.dv_expr.get())
            var_s   = self.dv_var.get().strip() or 'x'
            var     = sp.Symbol(var_s)
            ok, n   = validate_int(self.dv_order.get(), min_val=1)
            if not ok:
                raise ValueError(f"Orden inválido: {n}")
            pt_str  = self.dv_pt.get().strip()

            # Compute successive derivatives and show each step
            lines = [f"f({var_s}) = {self.dv_expr.get()}", ""]
            current = expr
            ordinals = ["primera", "segunda", "tercera", "cuarta", "quinta"]
            for i in range(1, n + 1):
                current = sp.diff(current, var)
                simplified = sp.simplify(current)
                expanded   = sp.expand(current)
                ord_name   = ordinals[i-1] if i <= len(ordinals) else f"orden {i}"
                prime = "'" * min(i, 3) + (f"({i})" if i > 3 else "")
                lines.append(f"f{prime}({var_s})  [{ord_name}]")
                lines.append(f"  = {current}")
                if simplified != current:
                    lines.append(f"  = {simplified}  (simplificada)")
                if expanded != current and expanded != simplified:
                    lines.append(f"  = {expanded}  (expandida)")
                lines.append("")

            # Evaluate at a point if provided
            if pt_str:
                pt_val = parse_math(pt_str)
                result_at_pt = current.subs(var, pt_val)
                simplified_pt = sp.simplify(result_at_pt)
                lines.append(f"Evaluada en {var_s} = {pt_str}:")
                lines.append(f"  f{'('+str(n)+')'}({pt_str}) = {simplified_pt}")
                try:
                    lines.append(f"  ≈ {float(simplified_pt.evalf()):.12g}")
                except:
                    pass

            self._show_text(self.dv_result, '\n'.join(lines))
            self._plot_deriv(expr, current, var, n, pt_str)
        except Exception as ex:
            self._show_text(self.dv_result, f"Error: {ex}")

    def _calc_integral(self):
        try:
            expr = parse_math(self.if_expr.get())
            var_s = self.if_var.get().strip() or 'x'
            var = sp.Symbol(var_s)
            indef = self.if_indef.get()

            if indef:
                result = sp.integrate(expr, var)
                simplified = sp.simplify(result)
                lines = [
                    f"∫ {self.if_expr.get()} d{var_s}",
                    f"= {result} + C",
                    f"= {simplified} + C  (simplificada)",
                ]
                try:
                    lines.append(f"Nota: resultado es función de {var_s}")
                except: pass
            else:
                a_str = self.if_a.get().strip()
                b_str = self.if_b.get().strip()
                a_expr = parse_math(a_str) if a_str else sp.Integer(0)
                b_expr = parse_math(b_str) if b_str else sp.oo

                result = sp.integrate(expr, (var, a_expr, b_expr))
                simplified = sp.simplify(result)
                try:
                    decimal = float(simplified.evalf())
                    dec_str = f"\n≈ {decimal:.12g}  (decimal)"
                except:
                    dec_str = "\n(resultado no numérico o complejo)"

                lines = [
                    f"∫ de {a_str} a {b_str}: {self.if_expr.get()} d{var_s}",
                    f"= {result}",
                    f"= {simplified}  (simplificada)" + dec_str,
                ]
                # Check convergence for improper
                if str(a_expr) == 'oo' or str(b_expr) == 'oo' or \
                   str(a_expr) == '-oo' or str(b_expr) == '-oo':
                    lines.append("(Integral impropia evaluada)")

            self._show_text(self.int_result, '\n'.join(lines))
            self._plot_integral(expr, var, indef,
                                a_expr if not indef else None,
                                b_expr if not indef else None)
        except Exception as ex:
            self._show_text(self.int_result, f"Error: {ex}")

    def _calc_limit(self):
        try:
            expr = parse_math(self.lm_expr.get())
            var_s = self.lm_var.get().strip() or 'x'
            var = sp.Symbol(var_s)
            pt_str = self.lm_pt.get().strip()
            pt = parse_math(pt_str)
            direction = self.lm_dir.get()

            if direction == 'bilateral':
                result = sp.limit(expr, var, pt)
                lines = [
                    f"lim({var_s}→{pt_str}) {self.lm_expr.get()}",
                    f"= {result}",
                    f"= {sp.simplify(result)}  (simplificado)",
                ]
                try:
                    lines.append(f"≈ {float(result.evalf()):.12g}")
                except: pass
                # Check left and right separately
                try:
                    rl = sp.limit(expr, var, pt, '-')
                    rr = sp.limit(expr, var, pt, '+')
                    if rl != rr:
                        lines.append(f"\n⚠ Lím. lateral izq = {rl}")
                        lines.append(f"   Lím. lateral der = {rr}")
                        lines.append(f"   → Límite bilateral NO existe")
                except: pass
            elif direction == 'right':
                result = sp.limit(expr, var, pt, '+')
                lines = [
                    f"lim({var_s}→{pt_str}⁺) {self.lm_expr.get()}",
                    f"= {result}",
                ]
            else:
                result = sp.limit(expr, var, pt, '-')
                lines = [
                    f"lim({var_s}→{pt_str}⁻) {self.lm_expr.get()}",
                    f"= {result}",
                ]
            try:
                lines.append(f"≈ {float(result.evalf()):.12g}")
            except: pass

            self._show_text(self.lm_result, '\n'.join(lines))
            self._plot_limit(expr, var, pt)
        except Exception as ex:
            self._show_text(self.lm_result, f"Error: {ex}")

    def _plot_deriv(self, expr, dexpr, var, n, pt_str):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        f  = sp.lambdify(var, expr,  modules=['numpy', {'Abs': np.abs}])
        df = sp.lambdify(var, dexpr, modules=['numpy', {'Abs': np.abs}])
        xs = np.linspace(-4, 4, 500)
        colors = ['#2a6ebb', '#e05c00', '#009944', '#9b00cc', '#bb2200']
        try:
            ys = np.real(np.array(f(xs), dtype=complex))
            ax.plot(xs, np.clip(ys, -50, 50), color=colors[0], lw=2, label='f(x)')
        except: pass
        try:
            yds = np.real(np.array(df(xs), dtype=complex))
            label = "f'(x)" if n == 1 else f"f({'('+str(n)+')'})(x)"
            ax.plot(xs, np.clip(yds, -50, 50), color=colors[min(n, 4)],
                    lw=1.8, ls='--', label=label)
        except: pass
        # Mark evaluation point
        if pt_str:
            try:
                xp = float(parse_math(pt_str).evalf())
                yp = float(sp.lambdify(var, dexpr, 'numpy')(xp))
                ax.scatter([xp], [yp], color='red', s=80, zorder=6,
                           label=f"f'({pt_str}) = {yp:.6g}")
                ax.axvline(xp, color='red', ls=':', lw=1)
            except: pass
        ax.axhline(0, color='k', lw=0.7)
        ax.set_title(f'f(x) y su derivada de orden {n}', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()

    def _plot_integral(self, expr, var, indef, a, b):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        f = sp.lambdify(var, expr, modules=['numpy', {'Abs': np.abs}])
        try:
            if indef or a is None:
                xs = np.linspace(-5, 5, 400)
            else:
                try:
                    af, bf = float(a.evalf()), float(b.evalf())
                    margin = max(1.0, (bf-af)*0.3)
                    xs = np.linspace(af - margin, bf + margin, 400)
                except:
                    xs = np.linspace(-5, 5, 400)
            ys = np.array([complex(f(x)) for x in xs])
            ys_r = np.real(ys)
            ax.plot(xs, ys_r, 'b-', lw=2, label='f(x)')
            if not indef and a is not None:
                try:
                    af2, bf2 = float(a.evalf()), float(b.evalf())
                    x_fill = np.linspace(af2, bf2, 300)
                    y_fill = np.array([float(f(x).real) for x in x_fill])
                    ax.fill_between(x_fill, 0, y_fill, alpha=0.3, color='blue', label='Área')
                    ax.axvline(af2, color='green', ls='--', lw=1)
                    ax.axvline(bf2, color='green', ls='--', lw=1)
                except: pass
        except: pass
        ax.axhline(0, color='k', lw=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_title('Función e integral', fontsize=9)
        ax.legend(fontsize=8)
        self.fig.tight_layout(); self.draw()

    def _plot_limit(self, expr, var, pt):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        f = sp.lambdify(var, expr, modules=['numpy', {'Abs': np.abs}])
        try:
            pt_f = float(pt.evalf())
            if abs(pt_f) > 1e10:
                xs = np.linspace(-20, 20, 400)
            else:
                margin = max(2.0, abs(pt_f)*0.5 + 1)
                xs = np.linspace(pt_f - margin, pt_f + margin, 400)
                # Remove singular points
                xs = xs[np.abs(xs - pt_f) > 1e-8]
            ys = []
            for x in xs:
                try:
                    v = complex(f(x))
                    ys.append(v.real if abs(v.imag) < 1e-10 else np.nan)
                except:
                    ys.append(np.nan)
            ys = np.array(ys)
            mask = np.abs(ys) < 1e4
            ax.plot(xs[mask], ys[mask], 'b-', lw=2)
            # Mark limit point
            try:
                lim_val = float(sp.limit(expr, var, pt).evalf())
                ax.scatter([pt_f], [lim_val], color='red', s=100, zorder=5,
                           label=f'Límite = {lim_val:.6g}')
                ax.legend(fontsize=8)
            except: pass
        except: pass
        ax.axhline(0, color='k', lw=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_title('Función cerca del punto límite', fontsize=9)
        self.fig.tight_layout(); self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 17 – SISTEMAS AUTÓNOMOS
# ══════════════════════════════════════════════════════════════════════════════
class TabSistemasAutonomos(TabBase):
    _FLINES = [
        r'$\dot{x} = f(x)\quad\text{(1D)}$',
        r'$\dot{x}=f(x,y),\ \dot{y}=g(x,y)\quad\text{(2D)}$',
    ]
    _DESC = (
        "Sistemas autónomos: la variable independiente no aparece explícitamente. "
        "Se analizan puntos de equilibrio (f(x*)=0), estabilidad lineal (signo de f'(x*) en 1D, "
        "autovalores del Jacobiano en 2D), y se dibujan diagramas de fase y campos directores."
    )

    def __init__(self, parent):
        super().__init__(parent, '⚡ Sistemas Autónomos — Diagramas de Fase')
        lp = self.lp
        formula_card(lp, self._FLINES, self._DESC)

        # Mode toggle 1D / 2D
        section(lp, 'Dimensión del sistema')
        self.dim_var = tk.StringVar(value='1d')
        mf = tk.Frame(lp, bg=PANEL); mf.pack(anchor='w', padx=6, pady=(4,0))
        tk.Radiobutton(mf, text='1D — Ecuación escalar dx/dt = f(x)', variable=self.dim_var,
                       value='1d', bg=PANEL, font=(FF,9), command=self._toggle_dim).pack(side='left', padx=4)
        tk.Radiobutton(mf, text='2D — Sistema dx/dt, dy/dt', variable=self.dim_var,
                       value='2d', bg=PANEL, font=(FF,9), command=self._toggle_dim).pack(side='left', padx=4)

        # 1D frame
        self.frm_1d = tk.Frame(lp, bg=PANEL)
        section(self.frm_1d, 'Sistema 1D')
        self.e1d_f    = LEntry(self.frm_1d, 'f(x) = dx/dt =', '3*x - x**2')
        self.e1d_x0   = LEntry(self.frm_1d, 'x₀ inicial =', '0.5')
        self.e1d_xmin = LEntry(self.frm_1d, 'x mínimo (gráfico) =', '-1')
        self.e1d_xmax = LEntry(self.frm_1d, 'x máximo (gráfico) =', '5')
        self.e1d_tmax = LEntry(self.frm_1d, 't máximo (simulación) =', '5')
        for w in [self.e1d_f, self.e1d_x0, self.e1d_xmin, self.e1d_xmax, self.e1d_tmax]:
            w.pack(fill='x', padx=6, pady=1)

        # 2D frame
        self.frm_2d = tk.Frame(lp, bg=PANEL)
        section(self.frm_2d, 'Sistema 2D')
        self.e2d_f      = LEntry(self.frm_2d, 'f(x,y) = dx/dt =', '2*x - x**2 - x*y')
        self.e2d_g      = LEntry(self.frm_2d, 'g(x,y) = dy/dt =', '3*y - y**2 - x*y')
        self.e2d_xmin   = LEntry(self.frm_2d, 'x mínimo =', '-0.5')
        self.e2d_xmax   = LEntry(self.frm_2d, 'x máximo =', '3.5')
        self.e2d_ymin   = LEntry(self.frm_2d, 'y mínimo =', '-0.5')
        self.e2d_ymax   = LEntry(self.frm_2d, 'y máximo =', '3.5')
        self.e2d_tmax   = LEntry(self.frm_2d, 't máximo (trayectorias) =', '10')
        self.e2d_n_traj = LEntry(self.frm_2d, 'N trayectorias =', '8')
        for w in [self.e2d_f, self.e2d_g, self.e2d_xmin, self.e2d_xmax,
                  self.e2d_ymin, self.e2d_ymax, self.e2d_tmax, self.e2d_n_traj]:
            w.pack(fill='x', padx=6, pady=1)

        self.frm_1d.pack(fill='x', padx=4, pady=2)
        self.frm_2d.pack_forget()

        btn(lp, '▶  Analizar', self._run)
        section(lp, 'Equilibrios y estabilidad')
        self.tree = make_table(lp,
            ['Equilibrio', 'Tipo / Estabilidad', 'Autovalores'],
            [100, 180, 180])
        self.res_lbl = result_label(lp)

    def _toggle_dim(self):
        if self.dim_var.get() == '1d':
            self.frm_2d.pack_forget()
            self.frm_1d.pack(fill='x', padx=4, pady=2)
        else:
            self.frm_1d.pack_forget()
            self.frm_2d.pack(fill='x', padx=4, pady=2)

    def _run(self):
        if self.dim_var.get() == '1d':
            self._run_1d()
        else:
            self._run_2d()

    @staticmethod
    def _classify_2d(eigenvalues):
        evs = [complex(e) for e in eigenvalues]
        reals = [e.real for e in evs]
        imags = [e.imag for e in evs]
        if all(abs(i) < 1e-10 for i in imags):  # real eigenvalues
            if all(r < -1e-10 for r in reals): return "Nodo estable"
            elif all(r > 1e-10 for r in reals): return "Nodo inestable"
            elif reals[0]*reals[1] < 0: return "Punto silla (inestable)"
            else: return "No hiperbólico"
        else:  # complex
            avg_real = sum(reals)/len(reals)
            if avg_real < -1e-10: return "Espiral estable"
            elif avg_real > 1e-10: return "Espiral inestable"
            else: return "Centro (neutralmente estable)"

    def _rk4_step(self, f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    def _run_1d(self):
        try:
            x_sym = sp.Symbol('x')
            expr = parse_math(self.e1d_f.get())
            f_np = to_f(expr)
            x0   = fval(self.e1d_x0.get())
            xmin = fval(self.e1d_xmin.get())
            xmax = fval(self.e1d_xmax.get())
            tmax = fval(self.e1d_tmax.get())

            # Find equilibria symbolically
            try:
                eq_sols = sp.solve(expr, x_sym)
            except Exception:
                eq_sols = []

            # Also find numerically in range
            xtest = np.linspace(xmin, xmax, 500)
            ftest = np.array([float(f_np(x)) for x in xtest])
            sign_changes = np.where(np.diff(np.sign(ftest)))[0]
            for idx in sign_changes:
                try:
                    from scipy.optimize import brentq
                    root = brentq(lambda x: float(f_np(x)), xtest[idx], xtest[idx+1])
                    # check if already in eq_sols
                    if not any(abs(float(complex(s).real) - root) < 1e-6 for s in eq_sols if s.is_real):
                        eq_sols.append(sp.Float(root))
                except Exception:
                    pass

            # Compute stability for each equilibrium
            d1expr = sp.diff(expr, x_sym)
            d1f = to_f(d1expr)
            equilibria = []
            for sol in eq_sols:
                try:
                    xval = float(complex(sol.evalf()).real)
                    if not (xmin - 1 <= xval <= xmax + 1): continue
                    fp = float(d1f(xval))
                    if fp < -1e-10:
                        stab = "Estable (atractor)"
                    elif fp > 1e-10:
                        stab = "Inestable (repulsor)"
                    else:
                        stab = "Semistable (indeterminado)"
                    equilibria.append((xval, stab, fp))
                except Exception:
                    pass

            table_clear(self.tree)
            for xval, stab, fp in equilibria:
                table_insert(self.tree, (f"x* = {xval:.6g}", stab, f"f'(x*) = {fp:.6g}"), '')

            n_msg = len(equilibria)
            self.res_lbl.config(text=f"Encontrados {n_msg} puntos de equilibrio en [{xmin}, {xmax}]", fg=OK)

            # Simulate solution curves with RK4
            h_sim = tmax / 500
            ic_list = [x0] + [xmin + (xmax-xmin)*i/6 for i in range(7)]
            curves = []
            for ic in ic_list:
                ts_sim = [0.0]
                ys_sim = [ic]
                for _ in range(500):
                    y_cur = ys_sim[-1]
                    try:
                        fval_cur = float(f_np(y_cur))
                        y_new = y_cur + h_sim * fval_cur
                        # simple euler for 1D
                        ys_sim.append(y_new)
                        ts_sim.append(ts_sim[-1] + h_sim)
                    except Exception:
                        break
                curves.append((ts_sim, ys_sim))

            self._plot_1d(f_np, equilibria, xmin, xmax, tmax, curves)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _plot_1d(self, f_np, equilibria, xmin, xmax, tmax, curves):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)

        # Phase line / portrait
        xs = np.linspace(xmin, xmax, 400)
        fs = np.array([float(f_np(x)) for x in xs])
        ax1.axhline(0, color='k', lw=1)
        ax1.plot(xs, fs, 'b-', lw=2, label='f(x)')
        ax1.fill_between(xs, 0, fs, where=(fs > 0), alpha=0.15, color='green', label='dx/dt > 0 (→)')
        ax1.fill_between(xs, 0, fs, where=(fs < 0), alpha=0.15, color='red', label='dx/dt < 0 (←)')
        colors_eq = {'Estable (atractor)': 'green', 'Inestable (repulsor)': 'red',
                     'Semistable (indeterminado)': 'orange'}
        for xval, stab, fp in equilibria:
            c = colors_eq.get(stab, 'purple')
            ax1.scatter([xval], [0], color=c, s=80, zorder=5)
            ax1.annotate(f'x*={xval:.3g}\n({stab[:6]})', (xval, 0),
                         textcoords='offset points', xytext=(0, 12), fontsize=7, ha='center', color=c)
        ax1.set_title('Diagrama de fase 1D — f(x) vs x', fontsize=9)
        ax1.set_xlabel('x'); ax1.set_ylabel('dx/dt = f(x)')
        ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

        # Solution curves x(t)
        for ts_sim, ys_sim in curves:
            ys_clipped = np.clip(ys_sim, xmin - 0.5, xmax + 0.5)
            ax2.plot(ts_sim, ys_clipped, lw=1.2, alpha=0.7)
        for xval, stab, fp in equilibria:
            c = colors_eq.get(stab, 'purple')
            ax2.axhline(xval, color=c, ls='--', lw=1, alpha=0.6)
        ax2.set_title('Curvas de solución x(t) desde distintas CIs', fontsize=9)
        ax2.set_xlabel('t'); ax2.set_ylabel('x(t)')
        ax2.set_xlim(0, tmax); ax2.set_ylim(xmin - 0.5, xmax + 0.5)
        ax2.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()

    def _run_2d(self):
        try:
            x_sym, y_sym = sp.Symbol('x'), sp.Symbol('y')
            expr_f = parse_math(self.e2d_f.get(), extra={'x': x_sym, 'y': y_sym})
            expr_g = parse_math(self.e2d_g.get(), extra={'x': x_sym, 'y': y_sym})
            f_np = to_f2(expr_f, 'x', 'y')
            g_np = to_f2(expr_g, 'x', 'y')
            xmin = fval(self.e2d_xmin.get()); xmax = fval(self.e2d_xmax.get())
            ymin = fval(self.e2d_ymin.get()); ymax = fval(self.e2d_ymax.get())
            tmax = fval(self.e2d_tmax.get())
            ok, n_traj = validate_int(self.e2d_n_traj.get(), 1)
            if not ok: raise ValueError(n_traj)

            # Find equilibria symbolically
            equilibria = []
            try:
                sols = sp.solve([expr_f, expr_g], [x_sym, y_sym], dict=True)
                for s in sols:
                    try:
                        xv = float(complex(s[x_sym].evalf()).real)
                        yv = float(complex(s[y_sym].evalf()).real)
                        if xmin-0.5 <= xv <= xmax+0.5 and ymin-0.5 <= yv <= ymax+0.5:
                            equilibria.append((xv, yv))
                    except Exception:
                        pass
            except Exception:
                pass

            # Numerical grid fallback if few equilibria found
            xg = np.linspace(xmin, xmax, 30)
            yg = np.linspace(ymin, ymax, 30)
            for xi in xg:
                for yi in yg:
                    try:
                        fv = abs(float(f_np(xi, yi)))
                        gv = abs(float(g_np(xi, yi)))
                        if fv < 0.05*(xmax-xmin) and gv < 0.05*(ymax-ymin):
                            # refine with simple iteration
                            if not any(abs(xi-e[0]) < 0.15*(xmax-xmin) and
                                       abs(yi-e[1]) < 0.15*(ymax-ymin) for e in equilibria):
                                equilibria.append((xi, yi))
                    except Exception:
                        pass

            # Compute Jacobian and eigenvalues at each equilibrium
            J_sym = sp.Matrix([[sp.diff(expr_f, x_sym), sp.diff(expr_f, y_sym)],
                                [sp.diff(expr_g, x_sym), sp.diff(expr_g, y_sym)]])
            eq_info = []
            for xv, yv in equilibria:
                try:
                    J_num = np.array([[float(complex(J_sym[0,0].subs([(x_sym, xv),(y_sym, yv)]).evalf()).real),
                                       float(complex(J_sym[0,1].subs([(x_sym, xv),(y_sym, yv)]).evalf()).real)],
                                      [float(complex(J_sym[1,0].subs([(x_sym, xv),(y_sym, yv)]).evalf()).real),
                                       float(complex(J_sym[1,1].subs([(x_sym, xv),(y_sym, yv)]).evalf()).real)]])
                    eigenvalues = np.linalg.eigvals(J_num)
                    stab = self._classify_2d(eigenvalues)
                    ev_str = ', '.join([f'{complex(e).real:.4g}{"+" if complex(e).imag>=0 else ""}{complex(e).imag:.4g}i'
                                        if abs(complex(e).imag) > 1e-10 else f'{complex(e).real:.4g}'
                                        for e in eigenvalues])
                    eq_info.append((xv, yv, stab, ev_str))
                except Exception:
                    eq_info.append((xv, yv, 'Error al calcular', '—'))

            table_clear(self.tree)
            for xv, yv, stab, ev_str in eq_info:
                table_insert(self.tree, (f"({xv:.4g}, {yv:.4g})", stab, ev_str), '')

            self.res_lbl.config(text=f"Encontrados {len(eq_info)} puntos de equilibrio", fg=OK)

            # Trajectories with RK4
            h_sim = min(0.02, tmax/200)
            n_steps = int(tmax / h_sim)
            # Generate ICs on a grid
            ic_xs = np.linspace(xmin + 0.1*(xmax-xmin), xmax - 0.1*(xmax-xmin), int(np.sqrt(n_traj))+1)
            ic_ys = np.linspace(ymin + 0.1*(ymax-ymin), ymax - 0.1*(ymax-ymin), int(np.sqrt(n_traj))+1)
            ics = [(xi, yi) for xi in ic_xs for yi in ic_ys][:n_traj]

            def deriv(t, state):
                xc, yc = state
                try:
                    return np.array([float(f_np(xc, yc)), float(g_np(xc, yc))])
                except Exception:
                    return np.array([0.0, 0.0])

            trajectories = []
            for ic_x, ic_y in ics:
                state = np.array([ic_x, ic_y])
                traj = [state.copy()]
                for _ in range(n_steps):
                    try:
                        state = self._rk4_step(deriv, 0, state, h_sim)
                        if not (xmin - 2 <= state[0] <= xmax + 2 and ymin - 2 <= state[1] <= ymax + 2):
                            break
                        traj.append(state.copy())
                    except Exception:
                        break
                trajectories.append(np.array(traj))

            self._plot_2d(f_np, g_np, eq_info, trajectories, xmin, xmax, ymin, ymax)
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    def _plot_2d(self, f_np, g_np, eq_info, trajectories, xmin, xmax, ymin, ymax):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Quiver field
        nx, ny = 18, 18
        xg = np.linspace(xmin, xmax, nx)
        yg = np.linspace(ymin, ymax, ny)
        XG, YG = np.meshgrid(xg, yg)
        UG = np.zeros_like(XG)
        VG = np.zeros_like(YG)
        for i in range(ny):
            for j in range(nx):
                try:
                    UG[i,j] = float(f_np(XG[i,j], YG[i,j]))
                    VG[i,j] = float(g_np(XG[i,j], YG[i,j]))
                except Exception:
                    pass
        speed = np.sqrt(UG**2 + VG**2)
        speed[speed == 0] = 1
        ax.quiver(XG, YG, UG/speed, VG/speed, speed, cmap='coolwarm', alpha=0.6, scale=25)

        # Trajectories
        colors_traj = matplotlib.cm.get_cmap('tab10')(np.linspace(0, 1, len(trajectories)))
        for traj, c in zip(trajectories, colors_traj):
            if len(traj) > 1:
                ax.plot(traj[:,0], traj[:,1], lw=1.2, color=c, alpha=0.8)
                ax.annotate('', xy=traj[min(len(traj)-1, len(traj)//2+1)],
                            xytext=traj[len(traj)//2],
                            arrowprops=dict(arrowstyle='->', color=c, lw=1.2))

        # Equilibria
        stab_colors = {'Nodo estable': 'green', 'Nodo inestable': 'red',
                       'Punto silla (inestable)': 'orange', 'Espiral estable': 'darkgreen',
                       'Espiral inestable': 'darkred', 'Centro (neutralmente estable)': 'blue',
                       'No hiperbólico': 'purple'}
        for xv, yv, stab, ev_str in eq_info:
            c = stab_colors.get(stab, 'black')
            marker = 'o' if 'estable' in stab.lower() and 'in' not in stab.lower() else (
                     's' if 'silla' in stab.lower() else 'D')
            ax.scatter([xv], [yv], color=c, s=100, zorder=6, marker=marker)
            ax.annotate(f'({xv:.2g},{yv:.2g})\n{stab[:12]}', (xv, yv),
                        textcoords='offset points', xytext=(8, 4), fontsize=7, color=c)

        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Campo director y trayectorias — Sistema 2D', fontsize=9)
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout(); self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Métodos Numéricos — Modelado y Simulación  |  Omar J. Cáceres 2026")
        self.geometry("1300x780")
        self.configure(bg=BG)
        self.minsize(900, 600)

        # Top banner
        banner = tk.Frame(self, bg=HDR, height=46)
        banner.pack(fill='x')
        banner.pack_propagate(False)
        tk.Label(banner,
                 text="FUNDAMENTOS DE MODELADO Y SIMULACIÓN — Métodos Numéricos",
                 bg=HDR, fg=HDR_FG, font=(FF, 13, 'bold')).pack(side='left', padx=14, pady=10)
        tk.Label(banner,
                 text="Omar J. Cáceres · UADE 2026",
                 bg=HDR, fg='#aac8f0', font=(FF, 9)).pack(side='right', padx=14)

        # Notebook
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', font=(FF, 8, 'bold'), padding=[8, 4],
                        background='#c8d8ec', foreground='#1e3a5f')
        style.map('TNotebook.Tab',
                  background=[('selected', ACCENT)],
                  foreground=[('selected', 'white')])

        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=4, pady=(0,4))

        tabs = [
            ("Bisección",       TabBiseccion),
            ("Punto Fijo",      TabPuntoFijo),
            ("Aitken",          TabAitken),
            ("Newton-Raphson",  TabNewtonRaphson),
            ("Lagrange",        TabLagrange),
            ("Dif. Finitas",    TabDifFinitas),
            ("Rectángulo",      TabRectangulo),
            ("Trapecio",        TabTrapecio),
            ("Simpson 1/3",     TabSimpson13),
            ("Simpson 3/8",     TabSimpson38),
            ("Monte Carlo",     TabMonteCarlo),
            ("Euler",           TabEuler),
            ("Heun",            TabHeun),
            ("RK4",             TabRK4),
            ("Comparador ODE",  TabComparador),
            ("Sist. Autónomos", TabSistemasAutonomos),
            ("🧮 Calculadora",  TabCalculadora),
        ]

        for name, cls in tabs:
            tab = cls(nb)
            nb.add(tab, text=f" {name} ")


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
