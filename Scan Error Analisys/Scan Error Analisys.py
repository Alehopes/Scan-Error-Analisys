import os
import tempfile
import numpy as np
import streamlit as st
import trimesh
import open3d as o3d
import cadquery as cq
import plotly.graph_objects as go

st.set_page_config(page_title="Metrología 3D: STL vs STEP", layout="wide")

st.title("Inspección Dimensional 3D (CAD STEP vs Escaneo STL)")
st.markdown("Carga el modelo nominal (STEP) y la nube/malla escaneada (STL) para calcular la desviación superficial y tolerancias.")

# --- Barra lateral: Parámetros y descargas ---
st.sidebar.header("Parámetros de Tolerancia")
tol_sup = st.sidebar.number_input("Tolerancia Superior (+mm)", value=0.20, step=0.05)
tol_inf = st.sidebar.number_input("Tolerancia Inferior (-mm)", value=-0.20, step=0.05)
icp_dist = st.sidebar.slider("Distancia máxima ICP (mm)", min_value=0.5, max_value=15.0, value=3.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("Descargas del Proyecto")

# Lectura del propio script y requirements para descarga web
current_script_path = os.path.abspath(__file__)
if os.path.exists(current_script_path):
    with open(current_script_path, "r", encoding="utf-8") as f:
        py_code = f.read()
    st.sidebar.download_button(
        label="📥 Descargar app.py",
        data=py_code,
        file_name="app.py",
        mime="text/x-python"
    )

req_text = """streamlit>=1.35.0
plotly>=5.20.0
trimesh>=4.3.0
open3d>=0.18.0
cadquery>=2.4.0
numpy>=1.26.0,<2.0.0
scipy>=1.13.0
"""
st.sidebar.download_button(
    label="📥 Descargar requirements.txt",
    data=req_text,
    file_name="requirements.txt",
    mime="text/plain"
)

# --- Zona de Carga de Archivos ---
col1, col2 = st.columns(2)
with col1:
    step_file = st.file_uploader("1. Modelo CAD Nominal (.step / .stp)", type=["step", "stp"])
with col2:
    stl_file = st.file_uploader("2. Malla / Escaneado Real (.stl)", type=["stl"])


def step_bytes_to_trimesh(file_bytes: bytes) -> trimesh.Trimesh:
    """Tesela el sólido STEP a una malla triangular densa usando OpenCASCADE."""
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f_step:
        f_step.write(file_bytes)
        step_path = f_step.name

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f_stl:
        tmp_stl_path = f_stl.name

    try:
        shape = cq.importers.importStep(step_path)
        cq.exporters.export(shape, tmp_stl_path, exportType="STL", tolerance=0.03, angularTolerance=0.1)
        cad_mesh = trimesh.load(tmp_stl_path)
    finally:
        if os.path.exists(step_path):
            os.remove(step_path)
        if os.path.exists(tmp_stl_path):
            os.remove(tmp_stl_path)
    return cad_mesh


if step_file and stl_file:
    if st.button("Ejecutar Alineación e Inspección", type="primary"):
        with st.spinner("1/4. Procesando geometrías y teselando STEP nominal..."):
            cad_mesh = step_bytes_to_trimesh(step_file.read())

            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f_stl_in:
                f_stl_in.write(stl_file.read())
                stl_in_path = f_stl_in.name

            scan_trimesh = trimesh.load(stl_in_path)
            os.remove(stl_in_path)

            if isinstance(scan_trimesh, trimesh.Trimesh):
                scan_pts, _ = trimesh.sample.sample_surface(scan_trimesh, min(len(scan_trimesh.vertices), 120_000))
            else:
                scan_pts = np.asarray(scan_trimesh.vertices)

            cad_pcd = o3d.geometry.PointCloud()
            cad_pcd.points = o3d.utility.Vector3dVector(cad_mesh.vertices)

            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(scan_pts)

        with st.spinner("2/4. Calculando traslación baricéntrica y ajuste ICP..."):
            # Prealineación por centroides de masas
            t_center = np.eye(4)
            t_center[:3, 3] = cad_pcd.get_center() - scan_pcd.get_center()
            scan_pcd.transform(t_center)

            cad_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.5, max_nn=30))
            scan_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.5, max_nn=30))

            reg = o3d.pipelines.registration.registration_icp(
                scan_pcd, cad_pcd, max_correspondence_distance=icp_dist,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=90)
            )
            scan_pcd.transform(reg.transformation)

        with st.spinner("3/4. Computando desviaciones superficiales con signo..."):
            aligned_pts = np.asarray(scan_pcd.points)
            closest_pts, dists, face_idx = cad_mesh.nearest.on_surface(aligned_pts)
            normals = cad_mesh.face_normals[face_idx]

            diff = aligned_pts - closest_pts
            signs = np.sign(np.sum(diff * normals, axis=1))
            signs[signs == 0] = 1.0
            errors = dists * signs

        st.success("Inspección dimensional completada con éxito.")

        # --- Métricas Estadísticas ---
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        median_err = np.median(errors)
        in_tol_pct = np.mean((errors >= tol_inf) & (errors <= tol_sup)) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Puntos evaluados", f"{len(errors):,}")
        m2.metric("Desviación media", f"{mean_err:+.3f} mm")
        m3.metric("Desv. Estándar (1σ)", f"{std_err:.3f} mm")
        m4.metric("Dentro de Tolerancia", f"{in_tol_pct:.1f}%")

        # --- Visualizaciones Gráficas ---
        g1, g2 = st.columns([1, 1])

        with g1:
            st.subheader("Distribución Estadística del Error")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=errors, nbinsx=120, name="Frecuencia",
                marker_color='#1f77b4', histnorm='probability density'
            ))
            fig_hist.add_vline(x=tol_sup, line_dash="dash", line_color="red", annotation_text=f"+Tol ({tol_sup}mm)")
            fig_hist.add_vline(x=tol_inf, line_dash="dash", line_color="blue", annotation_text=f"-Tol ({tol_inf}mm)")
            fig_hist.add_vline(x=mean_err, line_color="black", annotation_text=f"Media ({mean_err:+.2f}mm)")
            fig_hist.update_layout(
                xaxis_title="Desviación perpendicular al CAD (mm)",
                yaxis_title="Densidad",
                template="plotly_white",
                height=520
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with g2:
            st.subheader("Inspección Visual 3D (Colormap)")
            # Muestreo optimizado para rendering WebGL fluido
            sub_count = min(len(aligned_pts), 30_000)
            sub_idx = np.random.choice(len(aligned_pts), sub_count, replace=False)
            sub_pts = aligned_pts[sub_idx]
            sub_err = errors[sub_idx]

            v_lim = max(abs(tol_sup), abs(tol_inf)) * 1.5
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=sub_pts[:, 0], y=sub_pts[:, 1], z=sub_pts[:, 2],
                mode='markers',
                marker=dict(
                    size=2.2,
                    color=sub_err,
                    colorscale='Jet',
                    cmin=-v_lim,
                    cmax=v_lim,
                    colorbar=dict(title="Error (mm)"),
                    opacity=0.85
                )
            )])
            fig_3d.update_layout(
                scene=dict(aspectmode='data'),
                margin=dict(l=0, r=0, b=0, t=0),
                height=520
            )
            st.plotly_chart(fig_3d, use_container_width=True)