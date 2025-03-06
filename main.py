import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QSlider, QDoubleSpinBox, QGroupBox)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from voigt import voigt

class VoigtSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.controls = []  # 初始化控件列表
        self.initUI()
        
    def initUI(self):
        # 创建图形窗口
        self.plot = pg.PlotWidget()
        self.plot.setLabel('left', 'Intensity')
        self.plot.setLabel('bottom', 'Wavenumber (cm-1)')
        
        # 设置绘图风格
        self.plot.setBackground('w')
        for axis in ['left', 'bottom']:
            self.plot.getAxis(axis).setPen(pg.mkPen('k'))
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        # 创建范围设置控件
        range_group = QGroupBox("Plot Range")
        range_layout = QHBoxLayout()
        
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(1000, 4000)
        self.min_spin.setValue(3600)
        self.min_spin.setSingleStep(10)
        
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(1000, 4000)
        self.max_spin.setValue(3800)
        self.max_spin.setSingleStep(10)
        
        range_layout.addWidget(QLabel("Min:"))
        range_layout.addWidget(self.min_spin)
        range_layout.addWidget(QLabel("Max:"))
        range_layout.addWidget(self.max_spin)
        range_group.setLayout(range_layout)
        
        # 创建峰参数控件
        control_layout = QHBoxLayout()
        for i in range(2):  # 两个峰
            layout = self.create_peak_controls(i+1)
            control_layout.addLayout(layout)
        
        # 布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(range_group)
        main_layout.addWidget(self.plot)
        main_layout.addLayout(control_layout)
        
        self.setLayout(main_layout)
        self.setWindowTitle('Voigt Profile Simulator')
        
        # 连接信号
        self.min_spin.valueChanged.connect(self.update_range)
        self.max_spin.valueChanged.connect(self.update_range)
        self.update_plot()
        
    def create_peak_controls(self, peak_num):
        layout = QVBoxLayout()
        
        # 中心位置
        pos_group = QGroupBox(f'Peak {peak_num} Position')
        pos_layout = QVBoxLayout()
        
        pos_spin = QDoubleSpinBox()
        pos_spin.setRange(3600, 3800)
        pos_spin.setValue(3700)
        pos_spin.setSingleStep(0.1)
        
        pos_slider = QSlider()
        pos_slider.setOrientation(Qt.Orientation.Horizontal)  # 垂直滑块
        pos_slider.setRange(3600, 3800)
        pos_slider.setValue(3700)
        
        # 同步滑块和输入框
        pos_spin.valueChanged.connect(lambda val: pos_slider.setValue(int(val)))
        pos_slider.valueChanged.connect(pos_spin.setValue)
        pos_spin.valueChanged.connect(self.update_plot)
        
        pos_layout.addWidget(pos_spin)
        pos_layout.addWidget(pos_slider)
        pos_group.setLayout(pos_layout)
        
        # 强度
        int_label = QLabel(f'Peak {peak_num} Intensity:')
        int_spin = QDoubleSpinBox()
        int_spin.setRange(-10, 10)
        int_spin.setValue(0.5)
        int_spin.setSingleStep(0.01)
        int_spin.valueChanged.connect(self.update_plot)
        
        # 洛伦兹宽度
        lw_label = QLabel(f'Peak {peak_num} Lorentzian FWHM:')
        lw_spin = QDoubleSpinBox()
        lw_spin.setRange(0, 1000)
        lw_spin.setValue(10.0)
        lw_spin.setSingleStep(0.1)
        lw_spin.valueChanged.connect(self.update_plot)
        
        # 高斯宽度
        gw_label = QLabel(f'Peak {peak_num} Gaussian FWHM:')
        gw_spin = QDoubleSpinBox()
        gw_spin.setRange(0, 1000)
        gw_spin.setValue(0.0)
        gw_spin.setSingleStep(0.1)
        gw_spin.valueChanged.connect(self.update_plot)
        
        # 添加到布局
        layout.addWidget(pos_group)
        layout.addWidget(int_label)
        layout.addWidget(int_spin)
        layout.addWidget(lw_label)
        layout.addWidget(lw_spin)
        layout.addWidget(gw_label)
        layout.addWidget(gw_spin)
        
        # 保存控件引用
        self.controls.append({
            'position': pos_spin,
            'intensity': int_spin,
            'lorentz_width': lw_spin,
            'gauss_width': gw_spin,
            'slider': pos_slider
        })
        
        return layout
        
    def update_range(self):
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()
        
        # 更新滑块范围
        for controls in self.controls:
            controls['position'].setRange(min_val, max_val)
            controls['slider'].setRange(int(min_val), int(max_val))
        
        self.update_plot()
        
    def analyze_peaks(self, x, y):
        """分析光谱峰，返回峰位置和FWHM"""
        from scipy.signal import find_peaks
        from scipy.interpolate import interp1d
        
        # 找峰
        peaks, _ = find_peaks(y, height=0.1*max(y))
        peak_info = []
        
        for peak in peaks:
            # 计算半高全宽
            half_max = y[peak]/2
            
            # 找到左右交叉点的索引
            left_idx = np.where(y[:peak] <= half_max)[0][-1]
            right_idx = np.where(y[peak:] <= half_max)[0][0] + peak
            
            # 使用更精确的插值方法
            def find_crossing(x1, x2, y1, y2):
                """使用线性插值找到精确的交叉点"""
                return x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            
            # 计算左交叉点
            x_left = find_crossing(
                x[left_idx], x[left_idx+1],
                y[left_idx], y[left_idx+1]
            )
            
            # 计算右交叉点
            x_right = find_crossing(
                x[right_idx-1], x[right_idx],
                y[right_idx-1], y[right_idx]
            )
            
            fwhm = x_right - x_left
            
            peak_info.append({
                'position': x[peak],
                'fwhm': fwhm,
                'intensity': y[peak]
            })
            
        return peak_info
        
    def update_peak_info(self, peak_info):
        """更新峰信息显示"""
        if not hasattr(self, 'info_label'):
            # 创建结果显示区域
            self.info_group = QGroupBox("Peak Analysis")
            self.info_layout = QVBoxLayout()
            self.info_label = QLabel()
            self.info_label.setWordWrap(True)
            self.info_layout.addWidget(self.info_label)
            self.info_group.setLayout(self.info_layout)
            self.layout().insertWidget(1, self.info_group)
            
        # 更新显示内容
        text = ""
        for i, info in enumerate(peak_info):
            text += f"Peak {i+1}:\n"
            text += f"  Position: {info['position']:.2f} cm-1\n"
            text += f"  FWHM: {info['fwhm']:.2f} cm-1\n"
            text += f"  Intensity: {info['intensity']:.4f}\n\n"
        self.info_label.setText(text)
        
    def update_plot(self):
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()
        x = np.linspace(min_val, max_val, 1000)
        y = np.zeros_like(x, dtype=complex)  # 使用复数数组
        
        for controls in self.controls:
            pos = controls['position'].value()
            intensity = controls['intensity'].value()
            lw = controls['lorentz_width'].value()
            gw = controls['gauss_width'].value()
            
            if gw == 0:
                # 复数形式的洛伦兹线型
                y += intensity * (lw/2) / (x - pos + 1j*(lw/2))
            else:
                y += voigt(x, pos, intensity, lw, gw)
        
        # 对复数结果取模平方
        y = np.abs(y)**2
        
        self.plot.clear()
        self.plot.plot(x, y, pen='k')
        
        # 峰识别和计算
        peak_info = self.analyze_peaks(x, y)
        self.update_peak_info(peak_info)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VoigtSimulator()
    window.show()
    sys.exit(app.exec())
