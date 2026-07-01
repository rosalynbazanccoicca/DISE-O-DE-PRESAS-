# -*- coding: utf-8 -*-
"""
Software de Diseño y Verificación de Presas de Gravedad
=========================================================
Basado en los apuntes del curso de Mecánica de Fluidos:
  - Análisis a presa vacía (peso propio)
  - Análisis a presa llena (con agua): vuelco, deslizamiento, excentricidad
  - Esfuerzos en el talón y en el pie de la presa
  - Subpresión (opcional) y peso del agua sobre el talud aguas arriba (opcional)

Cómo ejecutar:
    streamlit run app.py

Esto abre automáticamente una pestaña en tu navegador (http://localhost:8501)
con una ventana interactiva: la gráfica y los resultados se recalculan
en tiempo real cada vez que cambias un valor.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# Configuración general de la página
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Diseño de Presas de Gravedad",
    page_icon="🏞️",
    layout="wide",
)

# ----------------------------------------------------------------------
# Funciones de geometría (polígono genérico -> área, centroide)
# ----------------------------------------------------------------------
def propiedades_poligono(puntos):
    """Área y centroide de un polígono cerrado (lista de tuplas x,y)
    usando el método del Shoelace. Los puntos deben ir en orden
    (horario o antihorario) recorriendo el contorno de la sección."""
    n = len(puntos)
    A = 0.0
    Cx = 0.0
    Cy = 0.0
    for i in range(n):
        x0, y0 = puntos[i]
        x1, y1 = puntos[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross
    A *= 0.5
    if abs(A) < 1e-9:
        return 0.0, 0.0, 0.0
    Cx /= (6 * A)
    Cy /= (6 * A)
    return abs(A), Cx, Cy


def centroide_trapecio(h1, h2, B):
    """Centroide (medido desde el lado de altura h1, es decir x=0)
    de un diagrama trapezoidal de alturas h1 (en x=0) y h2 (en x=B)."""
    if (h1 + h2) <= 0:
        return B / 2
    return B * (h1 + 2 * h2) / (3 * (h1 + h2))


# ----------------------------------------------------------------------
# Generadores de perfiles (modelos) de presa
# ----------------------------------------------------------------------
def perfil_triangular(H, m_abajo):
    """Perfil triangular simple: talud aguas arriba vertical,
    talud aguas abajo con relación m (horizontal : vertical)."""
    B = m_abajo * H
    puntos = [(0, 0), (0, H), (B, 0)]
    return puntos, B


def perfil_trapezoidal(H, b_corona, m_arriba, m_abajo):
    """Perfil trapezoidal con corona, talud aguas arriba y aguas abajo."""
    x_corona_izq = m_arriba * H
    x_corona_der = x_corona_izq + b_corona
    B = x_corona_der + m_abajo * H
    puntos = [(0, 0), (x_corona_izq, H), (x_corona_der, H), (B, 0)]
    return puntos, B


def perfil_compuesto_ejemplo(H_total=3.0):
    """Perfil escalonado de referencia (similar al ejemplo trabajado en
    clase: corona + cuerpo + talud triangular). Solo es un punto de
    partida; puedes editarlo manualmente con la opción 'Personalizada'."""
    puntos = [
        (0.0, 0.0),
        (0.0, H_total),
        (0.9, H_total),
        (0.9, H_total - 0.93),
        (4.0, 0.0),
    ]
    B = 4.0
    return puntos, B


def parsear_poligono_custom(texto):
    """Convierte un texto tipo '0,0; 0,3; 1,3; 4,0' en lista de puntos.
    Se espera que el primer punto sea el talón (base izquierda, y=0) y
    el último el pie (base derecha, y=0)."""
    puntos = []
    for par in texto.split(";"):
        par = par.strip()
        if not par:
            continue
        x_str, y_str = par.split(",")
        puntos.append((float(x_str), float(y_str)))
    if len(puntos) < 3:
        raise ValueError("Se necesitan al menos 3 puntos para formar un polígono.")
    x0, y0 = puntos[0]
    xn, yn = puntos[-1]
    if abs(y0) > 1e-6 or abs(yn) > 1e-6:
        raise ValueError("El primer y el último punto deben tener y = 0 (la base de la presa).")
    B = abs(xn - x0)
    return puntos, B


# ----------------------------------------------------------------------
# Núcleo del cálculo estructural (estabilidad de la presa)
# ----------------------------------------------------------------------
def calcular_caso(puntos, B, L, gamma_c, gamma_w, hw, considerar_subpresion,
                   factor_reduccion_subpresion, h_cola, considerar_agua_talud,
                   m_arriba, fs_min_volteo, fs_min_deslizamiento, mu, cohesion):
    """Calcula peso, fuerzas, factores de seguridad y esfuerzos para un
    caso dado (hw = 0 -> presa vacía; hw > 0 -> presa con agua)."""

    A_poligono, Cx, Cy = propiedades_poligono(puntos)
    W_concreto = gamma_c * A_poligono * L

    fuerzas_verticales = [
        {"nombre": "Peso propio de la presa (W)", "valor": W_concreto, "x": Cx, "signo": 1}
    ]

    # Peso del agua que se apoya sobre el talud aguas arriba (si aplica)
    W_agua_talud = 0.0
    if considerar_agua_talud and m_arriba and m_arriba > 0 and hw > 0:
        h_efectiva = min(hw, max(p[1] for p in puntos))
        area_cuna = 0.5 * m_arriba * h_efectiva ** 2
        W_agua_talud = gamma_w * area_cuna * L
        x_cuna = (m_arriba * h_efectiva) / 3.0
        fuerzas_verticales.append(
            {"nombre": "Peso del agua sobre el talud aguas arriba", "valor": W_agua_talud, "x": x_cuna, "signo": 1}
        )

    # Empuje hidrostático horizontal aguas arriba (lado del talón, empuja hacia el pie)
    FH = 0.0
    y_FH = 0.0
    if hw > 0:
        FH = gamma_w * hw ** 2 / 2.0 * L
        y_FH = hw / 3.0

    # Empuje hidrostático horizontal aguas abajo / de cola (lado del pie, empuja
    # hacia el talón). Es una fuerza real, independiente de la subpresión.
    FH2 = 0.0
    y_FH2 = 0.0
    if h_cola > 0:
        FH2 = gamma_w * h_cola ** 2 / 2.0 * L
        y_FH2 = h_cola / 3.0

    FH_neto = FH - FH2  # fuerza horizontal neta que tiende a deslizar/voltear la presa

    # Subpresión (uplift)
    U = 0.0
    x_U = B / 2
    if hw > 0 and considerar_subpresion:
        U = factor_reduccion_subpresion * gamma_w * ((hw + h_cola) / 2.0) * B * L
        x_U = centroide_trapecio(hw, h_cola, B)
        fuerzas_verticales.append({"nombre": "Subpresión (U)", "valor": U, "x": x_U, "signo": -1})

    # Resultante vertical neta
    N = sum(f["valor"] * f["signo"] for f in fuerzas_verticales)

    # Momentos respecto al PIE (x = B) de las fuerzas verticales (resistentes)
    M_resistente = sum(f["valor"] * f["signo"] * (B - f["x"]) for f in fuerzas_verticales)
    # Momento de volteo neto respecto al pie: el empuje aguas arriba voltea,
    # el empuje de la cola (aguas abajo) resiste el volteo.
    M_volcante = FH * y_FH - FH2 * y_FH2
    M_volcante_calc = max(M_volcante, 1e-9)

    # Factores de seguridad
    FS_volteo = (M_resistente / M_volcante_calc) if M_volcante > 1e-9 else float("inf")
    FR = N * mu + cohesion * B * L
    FS_deslizamiento = (FR / FH_neto) if FH_neto > 1e-9 else float("inf")

    # Ubicación de la resultante y excentricidad
    # IMPORTANTE: la excentricidad usada para los esfuerzos en la base (sigma)
    # se calcula SOLO con las fuerzas VERTICALES (peso propio, subpresión,
    # peso de agua sobre el talud), tal como en el método de clase (x'').
    # El empuje horizontal del agua (FH) NO se incluye aquí: ese momento
    # (M_volcante) se usa únicamente para el factor de seguridad al vuelco.
    if N > 1e-9:
        x_resultante = B - M_resistente / N
    else:
        x_resultante = B / 2
    e = x_resultante - B / 2  # positivo -> hacia el pie

    cumple_tercio_medio = abs(e) <= (B / 6) + 1e-9

    # Esfuerzos en la base (kgf/m2)
    sigma_prom = N / (B * L) if B * L > 0 else 0
    sigma_pie = sigma_prom * (1 + 6 * e / B) if B > 0 else 0
    sigma_talon = sigma_prom * (1 - 6 * e / B) if B > 0 else 0

    return {
        "A_poligono": A_poligono, "Cx": Cx, "Cy": Cy,
        "W_concreto": W_concreto, "W_agua_talud": W_agua_talud,
        "FH": FH, "y_FH": y_FH, "FH2": FH2, "y_FH2": y_FH2, "FH_neto": FH_neto,
        "U": U, "x_U": x_U,
        "N": N, "M_resistente": M_resistente, "M_volcante": M_volcante,
        "FS_volteo": FS_volteo, "FS_deslizamiento": FS_deslizamiento,
        "x_resultante": x_resultante, "e": e,
        "cumple_tercio_medio": cumple_tercio_medio,
        "sigma_pie": sigma_pie, "sigma_talon": sigma_talon,
        "fs_min_volteo": fs_min_volteo, "fs_min_deslizamiento": fs_min_deslizamiento,
        "fuerzas_verticales": fuerzas_verticales,
    }


# ----------------------------------------------------------------------
# Funciones de graficación (Plotly)
# ----------------------------------------------------------------------
def grafico_seccion(puntos, B, H, hw, h_cola, resultado, titulo):
    xs = [p[0] for p in puntos] + [puntos[0][0]]
    ys = [p[1] for p in puntos] + [puntos[0][1]]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xs, y=ys, fill="toself", mode="lines",
        line=dict(color="#5b7fb5", width=2),
        fillcolor="rgba(91,127,181,0.35)",
        name="Sección de la presa", hoverinfo="skip",
    ))

    if hw > 0:
        fig.add_trace(go.Scatter(
            x=[-0.6 * max(B, 1), 0], y=[hw, hw],
            mode="lines", line=dict(color="#2596be", width=2, dash="dash"),
            name=f"Nivel de agua aguas arriba (h={hw:.2f} m)",
        ))
        fig.add_shape(type="rect", x0=-0.6 * max(B, 1), x1=0, y0=0, y1=hw,
                      fillcolor="rgba(37,150,190,0.25)", line=dict(width=0))

    if h_cola > 0:
        fig.add_trace(go.Scatter(
            x=[B, B + 0.6 * max(B, 1)], y=[h_cola, h_cola],
            mode="lines", line=dict(color="#2bbd8e", width=2, dash="dash"),
            name=f"Nivel de agua aguas abajo / cola (h={h_cola:.2f} m)",
        ))
        fig.add_shape(type="rect", x0=B, x1=B + 0.6 * max(B, 1), y0=0, y1=h_cola,
                      fillcolor="rgba(43,189,142,0.25)", line=dict(width=0))

    cx = resultado["x_resultante"]
    fig.add_trace(go.Scatter(
        x=[cx], y=[0], mode="markers+text",
        marker=dict(size=10, color="#e35d5d"),
        text=["R"], textposition="top center",
        name="Ubicación de la resultante",
    ))

    fig.update_layout(
        title=titulo, xaxis_title="Ancho de la base B (m)", yaxis_title="Altura (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=420, margin=dict(l=40, r=20, t=50, b=40),
        showlegend=True, legend=dict(orientation="h", y=-0.2),
    )
    return fig


def grafico_esfuerzos(B, sigma_talon, sigma_pie):
    fig = go.Figure()
    xs = [0, B]
    ys = [sigma_talon, sigma_pie]

    fig.add_shape(type="rect", x0=B / 3, x1=2 * B / 3, y0=min(min(ys, [0]) or [0]) * 1.2 - 1,
                  y1=max(ys + [0]) * 1.2 + 1, fillcolor="rgba(230,200,90,0.18)", line=dict(width=0))

    colores = ["#e35d5d" if y < 0 else "#3fae6a" for y in ys]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line=dict(color="#5b7fb5", width=3),
        marker=dict(size=10, color=colores),
        fill="tozeroy", fillcolor="rgba(91,127,181,0.15)",
        name="Distribución de esfuerzos",
    ))
    fig.add_hline(y=0, line=dict(color="white", width=1, dash="dot"))

    fig.add_annotation(x=0, y=sigma_talon, text=f"Talón: {sigma_talon:,.1f} kgf/m²",
                        showarrow=True, arrowhead=2, ax=0, ay=-40)
    fig.add_annotation(x=B, y=sigma_pie, text=f"Pie: {sigma_pie:,.1f} kgf/m²",
                        showarrow=True, arrowhead=2, ax=0, ay=-40)

    fig.update_layout(
        title="Distribución de esfuerzos en la base",
        xaxis_title="Ancho de la base B (m)", yaxis_title="Esfuerzo σ (kgf/m²)",
        height=380, margin=dict(l=40, r=20, t=50, b=40), showlegend=False,
    )
    return fig


def tarjeta_estado(nombre, valor, minimo, unidad="", es_mayor_mejor=True, fmt="{:.3f}"):
    cumple = (valor >= minimo) if es_mayor_mejor else (valor <= minimo)
    color = "#3fae6a" if cumple else "#e35d5d"
    estado = "✔ CUMPLE" if cumple else "✘ NO CUMPLE"
    valor_txt = fmt.format(valor) if np.isfinite(valor) else "∞"
    st.markdown(
        f"""
        <div style="border:1px solid #444;border-radius:10px;padding:14px 16px;background:#1c1f26;">
            <div style="font-size:0.8rem;color:#9aa4b2;letter-spacing:0.05em;text-transform:uppercase;">{nombre}</div>
            <div style="font-size:1.8rem;font-weight:700;color:#e8edf2;margin:4px 0;">{valor_txt} {unidad}</div>
            <div style="font-size:0.78rem;color:#9aa4b2;">mín. {minimo}</div>
            <div style="font-size:0.85rem;font-weight:600;color:{color};margin-top:6px;">{estado}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# BARRA LATERAL: parámetros de entrada
# ----------------------------------------------------------------------
st.sidebar.title("🏞️ Parámetros de diseño")

st.sidebar.header("1. Modelo de geometría")
modelo = st.sidebar.selectbox(
    "Selecciona el perfil de la presa",
    ["Triangular simple", "Trapezoidal con corona", "Compuesto (ejemplo de clase)", "Personalizado (polígono)"],
)

m_arriba_global = 0.0  # para el peso de agua sobre talud, solo aplica en algunos modelos

if modelo == "Triangular simple":
    H = st.sidebar.number_input("Altura H (m)", 0.5, 200.0, 8.0, 0.5)
    m_abajo = st.sidebar.number_input("Talud aguas abajo (m, H:V)", 0.0, 5.0, 0.8, 0.05)
    puntos, B = perfil_triangular(H, m_abajo)

elif modelo == "Trapezoidal con corona":
    H = st.sidebar.number_input("Altura H (m)", 0.5, 200.0, 8.0, 0.5)
    b_corona = st.sidebar.number_input("Ancho de corona (m)", 0.1, 20.0, 1.0, 0.1)
    m_arriba_global = st.sidebar.number_input("Talud aguas arriba (m, H:V)", 0.0, 3.0, 0.1, 0.05)
    m_abajo = st.sidebar.number_input("Talud aguas abajo (m, H:V)", 0.0, 5.0, 0.75, 0.05)
    puntos, B = perfil_trapezoidal(H, b_corona, m_arriba_global, m_abajo)

elif modelo == "Compuesto (ejemplo de clase)":
    H = st.sidebar.number_input("Altura total H (m)", 1.0, 50.0, 3.0, 0.1)
    puntos, B = perfil_compuesto_ejemplo(H)

else:  # Personalizado
    st.sidebar.caption(
        "Ingresa los puntos del contorno de la presa como pares x,y separados "
        "por punto y coma, recorriendo la sección en orden. El primer y el "
        "último punto deben estar sobre la base (y = 0): talón → corona → pie."
    )
    texto_puntos = st.sidebar.text_area(
        "Puntos (x,y; x,y; ...)",
        value="0,0; 0,8; 1,8; 4.6,0",
        height=90,
    )
    try:
        puntos, B = parsear_poligono_custom(texto_puntos)
        H = max(p[1] for p in puntos)
    except Exception as ex:
        st.sidebar.error(f"Error en los puntos: {ex}")
        st.stop()

st.sidebar.header("2. Propiedades de materiales")
gamma_c = st.sidebar.number_input("Peso específico del concreto γc (kgf/m³)", 1500.0, 4000.0, 2300.0, 50.0)
gamma_w = st.sidebar.number_input("Peso específico del agua γw (kgf/m³)", 900.0, 1100.0, 1000.0, 10.0)
L = st.sidebar.number_input("Longitud de análisis L (m)", 0.1, 1000.0, 1.0, 0.1,
                             help="Tramo de presa que se analiza; usa 1 m para un análisis por metro lineal.")

st.sidebar.header("3. Nivel de agua")
hw = st.sidebar.slider("Nivel de embalse hw, aguas arriba (m)", 0.0, float(H), float(min(H, H * 0.9)), 0.05)
h_cola = st.sidebar.number_input(
    "Tirante de agua aguas abajo / de cola (m)", 0.0, float(H), 0.0, 0.1,
    help="Si hay agua del lado del pie de la presa (aguas abajo), genera un empuje "
         "horizontal real que resiste el volteo y el deslizamiento. Pon 0 si no hay.",
)

with st.sidebar.expander("Opciones adicionales (subpresión / agua sobre talud)"):
    considerar_subpresion = st.checkbox("Considerar subpresión (U)", value=False)
    factor_reduccion = st.slider("Factor de reducción por drenes (Cr)", 0.0, 1.0, 1.0, 0.05,
                                  disabled=not considerar_subpresion)
    considerar_agua_talud = st.checkbox(
        "Considerar peso del agua sobre el talud aguas arriba", value=False,
        disabled=(m_arriba_global == 0),
        help="Solo aplica si el perfil tiene talud aguas arriba inclinado (modelo Trapezoidal con corona).",
    )

st.sidebar.header("4. Fricción y cohesión en la base")
mu = st.sidebar.number_input("Coeficiente de fricción μ", 0.3, 1.0, 0.75, 0.01)
cohesion = st.sidebar.number_input("Cohesión / adherencia c (kgf/m²)", 0.0, 50000.0, 0.0, 500.0)

st.sidebar.header("5. Factores de seguridad mínimos")
fs_min_volteo = st.sidebar.number_input("FS mínimo de volteo", 1.0, 3.0, 1.5, 0.05)
fs_min_deslizamiento = st.sidebar.number_input("FS mínimo de deslizamiento", 1.0, 3.0, 1.3, 0.05)

st.sidebar.header("6. Capacidad portante del suelo (opcional)")
considerar_sigma_adm = st.sidebar.checkbox("Verificar contra esfuerzo admisible del suelo", value=False)
sigma_adm = st.sidebar.number_input(
    "σ admisible del suelo (kgf/m²)", 0.0, 1_000_000.0, 9500.0, 100.0,
    disabled=not considerar_sigma_adm,
    help="Esfuerzo de trabajo permitido por el suelo de fundación (de tu estudio de mecánica de suelos).",
)
st.sidebar.caption(
    "Nota: f'c (resistencia del concreto) no se usa aquí porque no afecta la estabilidad "
    "global de la presa (vuelco/deslizamiento/esfuerzos en el suelo); ese dato se usa para "
    "el diseño estructural del concreto, que es una verificación aparte."
)

# ----------------------------------------------------------------------
# CÁLCULOS
# ----------------------------------------------------------------------
res_sin_agua = calcular_caso(
    puntos, B, L, gamma_c, gamma_w, 0.0, False, 1.0, 0.0, False, m_arriba_global,
    fs_min_volteo, fs_min_deslizamiento, mu, cohesion,
)
res_con_agua = calcular_caso(
    puntos, B, L, gamma_c, gamma_w, hw, considerar_subpresion, factor_reduccion, h_cola,
    considerar_agua_talud, m_arriba_global, fs_min_volteo, fs_min_deslizamiento, mu, cohesion,
)

# ----------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------
st.title("🏞️ Diseño y Verificación de Presas de Gravedad")
st.caption(
    "Herramienta interactiva basada en el análisis de estabilidad visto en clase "
    "(presa vacía y presa llena: vuelco, deslizamiento, excentricidad y esfuerzos en la base). "
    "Todos los resultados se actualizan automáticamente al cambiar los valores."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Área de la sección", f"{res_sin_agua['A_poligono']:.2f} m²")
c2.metric("Ancho de base B", f"{B:.2f} m")
c3.metric("Peso de la presa (W)", f"{res_sin_agua['W_concreto']:,.0f} kgf")
c4.metric("Altura H", f"{H:.2f} m")

tab1, tab2, tab3 = st.tabs(["💧 Sin Agua", "🌊 Con Agua", "📊 Resumen y Solución"])

# ----------------------------------------------------------------------
# TAB 1: SIN AGUA
# ----------------------------------------------------------------------
with tab1:
    colA, colB = st.columns([1.2, 1])
    with colA:
        st.plotly_chart(
            grafico_seccion(puntos, B, H, 0.0, 0.0, res_sin_agua, "Sección de la presa — Caso SIN AGUA"),
            use_container_width=True,
            key="grafico_seccion_sin_agua",
        )
    with colB:
        st.plotly_chart(
            grafico_esfuerzos(B, res_sin_agua["sigma_talon"], res_sin_agua["sigma_pie"]),
            use_container_width=True,
            key="grafico_esfuerzos_sin_agua",
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Excentricidad |e|", f"{abs(res_sin_agua['e']):.4f} m",
                   help=f"Límite del tercio medio: B/6 = {B/6:.4f} m")
        st.write("✔ Dentro del tercio medio" if res_sin_agua["cumple_tercio_medio"] else "✘ Fuera del tercio medio")
    with col2:
        st.metric("σ Talón", f"{res_sin_agua['sigma_talon']:,.1f} kgf/m²",
                   f"{res_sin_agua['sigma_talon']/10000:.4f} kgf/cm²")
    with col3:
        st.metric("σ Pie", f"{res_sin_agua['sigma_pie']:,.1f} kgf/m²",
                   f"{res_sin_agua['sigma_pie']/10000:.4f} kgf/cm²")

    if res_sin_agua["sigma_talon"] < 0 or res_sin_agua["sigma_pie"] < 0:
        st.warning("⚠️ Existen esfuerzos de tracción (negativos) en la base. Revisa la geometría.")

# ----------------------------------------------------------------------
# TAB 2: CON AGUA
# ----------------------------------------------------------------------
with tab2:
    colA, colB = st.columns([1.2, 1])
    with colA:
        st.plotly_chart(
            grafico_seccion(puntos, B, H, hw, h_cola, res_con_agua, "Sección de la presa — Caso CON AGUA"),
            use_container_width=True,
            key="grafico_seccion_con_agua",
        )
    with colB:
        st.plotly_chart(
            grafico_esfuerzos(B, res_con_agua["sigma_talon"], res_con_agua["sigma_pie"]),
            use_container_width=True,
            key="grafico_esfuerzos_con_agua",
        )

    st.subheader("Factores de seguridad")
    ncols = 4 if considerar_sigma_adm else 3
    cols_fs = st.columns(ncols)
    with cols_fs[0]:
        tarjeta_estado("FS Volteo", res_con_agua["FS_volteo"], fs_min_volteo)
    with cols_fs[1]:
        tarjeta_estado("FS Deslizamiento", res_con_agua["FS_deslizamiento"], fs_min_deslizamiento)
    with cols_fs[2]:
        tarjeta_estado("Excentricidad |e| (m)", abs(res_con_agua["e"]), B / 6, es_mayor_mejor=False)
    if considerar_sigma_adm:
        with cols_fs[3]:
            sigma_max = max(res_con_agua["sigma_talon"], res_con_agua["sigma_pie"])
            tarjeta_estado("σ máx. vs σ adm. (kgf/m²)", sigma_max, sigma_adm, es_mayor_mejor=False, fmt="{:,.0f}")

    st.markdown("####")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Empuje hidrostático FH (aguas arriba)", f"{res_con_agua['FH']:,.0f} kgf")
    g2.metric("Empuje de cola FH2 (aguas abajo)", f"{res_con_agua['FH2']:,.0f} kgf")
    g3.metric("Subpresión U", f"{res_con_agua['U']:,.0f} kgf" if considerar_subpresion else "No considerada")
    g4.metric("σ Talón / σ Pie", f"{res_con_agua['sigma_talon']:,.0f} / {res_con_agua['sigma_pie']:,.0f} kgf/m²")

    with st.expander("Ver detalle de fuerzas verticales consideradas"):
        df_fuerzas = pd.DataFrame(res_con_agua["fuerzas_verticales"])
        df_fuerzas["valor (kgf)"] = df_fuerzas["valor"].map(lambda v: f"{v:,.1f}")
        df_fuerzas["x desde el talón (m)"] = df_fuerzas["x"].map(lambda v: f"{v:.3f}")
        st.dataframe(df_fuerzas[["nombre", "valor (kgf)", "x desde el talón (m)"]], use_container_width=True)

# ----------------------------------------------------------------------
# TAB 3: RESUMEN
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Comparación Sin Agua vs Con Agua")

    tabla = pd.DataFrame({
        "Parámetro": [
            "FS Volteo", "FS Deslizamiento", "Excentricidad e (m)",
            "σ Talón (kgf/m²)", "σ Pie (kgf/m²)", "¿Dentro del tercio medio?",
        ],
        "Sin Agua": [
            "—", "—", f"{res_sin_agua['e']:.4f}",
            f"{res_sin_agua['sigma_talon']:.1f}", f"{res_sin_agua['sigma_pie']:.1f}",
            "Sí" if res_sin_agua["cumple_tercio_medio"] else "No",
        ],
        "Con Agua": [
            f"{res_con_agua['FS_volteo']:.3f}" if np.isfinite(res_con_agua['FS_volteo']) else "∞",
            f"{res_con_agua['FS_deslizamiento']:.3f}" if np.isfinite(res_con_agua['FS_deslizamiento']) else "∞",
            f"{res_con_agua['e']:.4f}",
            f"{res_con_agua['sigma_talon']:.1f}", f"{res_con_agua['sigma_pie']:.1f}",
            "Sí" if res_con_agua["cumple_tercio_medio"] else "No",
        ],
    })
    st.table(tabla)

    cumple_general = (
        res_con_agua["FS_volteo"] >= fs_min_volteo
        and res_con_agua["FS_deslizamiento"] >= fs_min_deslizamiento
        and res_con_agua["cumple_tercio_medio"]
        and res_con_agua["sigma_talon"] >= 0
        and res_con_agua["sigma_pie"] >= 0
    )
    if considerar_sigma_adm:
        cumple_general = cumple_general and max(res_con_agua["sigma_talon"], res_con_agua["sigma_pie"]) <= sigma_adm

    if cumple_general:
        st.success("✅ La presa CUMPLE con todos los criterios de estabilidad evaluados.")
    else:
        st.error("❌ La presa NO CUMPLE con uno o más criterios de estabilidad. Ajusta la geometría o los parámetros.")

    st.caption(
        "Fórmulas: FSvolteo = ΣM resistentes / ΣM volcantes neto (respecto al pie) · "
        "FSdeslizamiento = (N·μ + c·B·L) / (FH − FH2) · "
        "e = x̄ − B/2 (criterio del tercio medio: |e| ≤ B/6) · "
        "σ = N/(B·L) · (1 ± 6e/B). FH2 es el empuje hidrostático del agua aguas abajo (de cola), "
        "si la hay; resiste el volteo y el deslizamiento."
    )
