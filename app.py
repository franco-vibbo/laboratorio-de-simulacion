"""Laboratorio interactivo de Econometría. Ejecutar con: streamlit run app.py"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from simulacion import BETA0, BETA1, resumen, simular_heterocedasticidad

st.set_page_config(page_title="Laboratorio de Econometría", layout="wide")


@st.cache_data
def correr(n, reps, gamma, sigma, semilla):
    return simular_heterocedasticidad(n, reps, gamma, sigma, semilla)


def grafico_muestra(res):
    x, y = res["x0"], res["y0"]
    xs = np.array([x.min(), x.max()])
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="markers", name="Datos",
                    marker=dict(size=6, opacity=0.55))
    fig.add_scatter(x=xs, y=BETA0 + BETA1 * xs, mode="lines", name="Recta verdadera",
                    line=dict(dash="dash", color="gray"))
    fig.add_scatter(x=xs, y=res["b0_0"] + res["b1_0"] * xs, mode="lines",
                    name="Recta MCO", line=dict(color="#eb6834"))
    fig.update_layout(height=360, margin=dict(t=10, b=10), xaxis_title="x", yaxis_title="y",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def grafico_histograma(res):
    fig = go.Figure(go.Histogram(x=res["b1"], nbinsx=40, marker_color="#2a78d6"))
    fig.add_vline(x=BETA1, line_dash="dash", line_color="gray",
                  annotation_text="verdadero", annotation_position="top")
    fig.update_layout(height=360, margin=dict(t=10, b=10), xaxis_title="β̂₁",
                      yaxis_title="Frecuencia", bargap=0.05)
    return fig


st.title("Laboratorio interactivo de Econometría")
modulo = st.sidebar.radio("Módulo", ["1. MCO base", "2. Heterocedasticidad"])

st.sidebar.markdown("---")
n = st.sidebar.slider("Tamaño muestral (n)", 20, 300, 100, step=10)
reps = st.sidebar.select_slider("Réplicas", options=[500, 1000, 2000, 5000], value=2000)
sigma = st.sidebar.slider("Desvío del error (σ)", 0.2, 2.0, 0.6, step=0.1)
semilla = st.sidebar.number_input("Semilla", value=11, step=1)

if modulo.startswith("1"):
    gamma = 0.0
    st.header("MCO base: todos los supuestos se cumplen")
    st.write("Punto de referencia: con errores homocedásticos, MCO es insesgado y los "
             "intervalos de confianza cubren el valor verdadero cerca del 95% de las veces.")
else:
    st.header("Heterocedasticidad")
    gamma = st.sidebar.slider("Heterocedasticidad (γ)", 0.0, 1.5, 0.8, step=0.1)
    st.write("La varianza del error crece con x. ¿Qué le pasa al estimador y a la inferencia?")

res = correr(n, reps, gamma, sigma, int(semilla))
r = resumen(res)

izq, der = st.columns(2)
izq.subheader("Una sola muestra")
izq.plotly_chart(grafico_muestra(res), width="stretch")
der.subheader(f"Las {reps} estimaciones de β̂₁")
der.plotly_chart(grafico_histograma(res), width="stretch")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sesgo", f"{r['sesgo']:+.3f}")
c2.metric("Dispersión de β̂₁", f"{r['dispersion']:.3f}")
c3.metric("Cobertura IC clásico", f"{r['cobertura_clasico']:.1%}")
c4.metric("Cobertura IC robusto", f"{r['cobertura_robusto']:.1%}")

with st.expander("¿Qué debería observar el alumno?"):
    if modulo.startswith("1"):
        st.markdown("- El histograma está centrado en el valor verdadero (sesgo ≈ 0).\n"
                    "- Al subir *n*, la dispersión de β̂₁ baja.\n"
                    "- Ambas coberturas rondan el 95%.")
    else:
        st.markdown("- El histograma **sigue centrado** en el valor verdadero: MCO sigue siendo insesgado.\n"
                    "- Al subir γ, la nube de puntos forma un embudo y la dispersión de β̂₁ aumenta.\n"
                    "- La cobertura del IC **clásico** cae por debajo del 95%; la del **robusto** se mantiene cerca.\n"
                    "- Subir *n* mejora la precisión, pero no corrige el error estándar clásico.")
