import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import rasterio
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import colors
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_geotiff_with_shapes(folder_path, output_folder):
    # 获取文件夹中所有的GeoTIFF文件
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    all_data_max = -np.inf  # 用于存储所有数据的最大值

    # 第一步：遍历所有GeoTIFF文件，计算数据中的最大值
    for tif_file in tif_files:
        file_path = os.path.join(folder_path, tif_file)
        with rasterio.open(file_path) as src:
            data = src.read(1)  # 读取第一个波段数据
            no_data_value = -9999  # 数据中无效值的标记
            data[data == no_data_value] = np.nan  # 将无效值替换为NaN
            data_max = np.nanmax(data)  # 计算该文件中的最大值
            if data_max > all_data_max:
                all_data_max = data_max  # 更新所有数据的最大值

    # 第二步：绘制每个GeoTIFF文件的图像
    for tif_file in tif_files:
        file_path = os.path.join(folder_path, tif_file)
        with rasterio.open(file_path) as src:
            bounds = src.bounds  # 获取文件的地理边界
            m = Basemap(
                llcrnrlon=bounds.left,
                llcrnrlat=bounds.bottom,
                urcrnrlon=bounds.right,
                urcrnrlat=bounds.top,
                resolution='i',  # 中等分辨率
                projection='cyl'  # 等距圆柱投影
            )

            # 读取数据并处理无效值
            data = src.read(1)
            data[data == no_data_value] = np.nan  # 将无效值替换为NaN
            data_masked = np.ma.masked_invalid(data)  # 使用masked数组忽略NaN值

            # 使用指定的颜色映射绘制数据
            cmap = plt.get_cmap('YlGnBu')
            norm = colors.Normalize(vmin=0, vmax=all_data_max)  # 设置颜色映射范围
            m.imshow(data_masked, cmap=cmap, norm=norm, alpha=0.7, zorder=5, origin='upper', interpolation='none')

            # 在图上添加年份标签（从文件名提取年份）
            year = tif_file.split('_')[0]
            plt.text(0.05, 0.9, year, fontsize=14, color='black', transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='white', boxstyle='round,pad=0.5'))

            # 添加颜色条
            cbar = plt.colorbar(shrink=0.5, pad=0.02)
            cbar.set_label('Data Values', fontsize=10)

            # 绘制胡焕庸线
            m.readshapefile(r'file1', 'hux', linewidth=2, color='darkred', zorder=9)
            m.readshapefile(r'file2', 'provience', drawbounds=True, linewidth=1, color='black', zorder=6)

            # 提取省份边界并修复无效几何
            province_shapes = [make_valid(ShapelyPolygon(shape).buffer(0)) for shape in m.provience]
            province_union = unary_union(province_shapes)

            # 创建胡焕庸线的五边形坐标
            vertices = [
                (127.5212, 50.24297),
                (127.5212, 60),
                (60, 60),
                (60, 25.02394),
                (98.48866, 25.02394),
                (127.5212, 50.24297)
            ]
            polygon_coords = np.array(vertices)
            hu_left_polygon = ShapelyPolygon(polygon_coords).buffer(0)

            # 计算胡焕庸线与省份边界的交集
            intersection = hu_left_polygon.intersection(province_union)

            # 如果交集是多个多边形，处理每个多边形
            if isinstance(intersection, MultiPolygon):
                intersection_patches = [Polygon(np.array(poly.exterior.coords)) for poly in intersection.geoms]
            else:
                intersection_patches = [Polygon(np.array(intersection.exterior.coords))]

            # 创建PatchCollection，设置为灰色填充，透明度为0.5
            pc = PatchCollection(intersection_patches, facecolor='lightgray', edgecolor='gray', linewidth=0.5, alpha=0.5, zorder=8)
            plt.gca().add_collection(pc)

            # 设置标题
            plt.title(f'({tif_file})', fontsize=12, pad=20)

            # 添加南海诸岛小地图
            ax_inset = inset_axes(
                plt.gca(),
                width="100%",
                height="100%",
                loc='lower right',
                bbox_to_anchor=(0.83, 0, 0.2, 0.2),
                bbox_transform=plt.gca().transAxes,
                borderpad=0
            )

            m_inset = Basemap(
                llcrnrlon=105, urcrnrlon=125,
                llcrnrlat=0, urcrnrlat=25,
                resolution='i', projection='cyl', ax=ax_inset
            )
            ax_inset.set_xlim(105, 125)
            ax_inset.set_ylim(3, 25)

            # 绘制南海诸岛边界
            m_inset.readshapefile(r'file3', 'south_china_sea_wgs84', linewidth=1, color='black', zorder=6)

            # 设置小地图的边界
            for spine in ax_inset.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)

            # 自动调整布局
            plt.tight_layout()

            # 保存图像
            output_filename = os.path.join(output_folder, os.path.splitext(tif_file)[0] + '.png')
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.show()  # 显示当前图像
            plt.close()  # 关闭当前图形，避免内存泄漏

    print(f"所有图像已保存到 {output_folder} 文件夹！")
