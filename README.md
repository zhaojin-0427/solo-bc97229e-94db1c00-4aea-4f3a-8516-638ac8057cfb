# 海洋温度跨日 NetCDF 子集提取服务

面向海洋研究数据工程师的 Web 服务：上传多个每日 NetCDF 归档，按
**时间 / 纬度 / 经度**闭区间提取温度子集，跨文件去重合并并输出**单个
排序后的 NetCDF 文件**。

## 功能与约束

- 输入文件仅含一维坐标 `time`、`lat`、`lon` 与变量
  `temperature(time, lat, lon)`；各文件格点集合相同，**纬度顺序可相反**。
- `time` 为严格递增的 Unix 整秒；`lon` 位于 `[0, 360)` 且严格递增；
  `lat` 严格单调（升序或降序）。
- 经度选区以 `[-180, 180)` 表示的闭区间 `[west, east]`：
  - `west <= east`：普通区间；
  - `west > east`：**跨越日期变更线**（如 `[170, -170]`）。
- 输出坐标顺序：**time 升序 → lat 升序 → 转换后的 lon 升序**，经度由
  `[0,360)` 转换到 `[-180,180)`。
- 同一时刻、同一格点的多条记录：
  - 两个有效值必须**完全相等**，相等则合一；
  - **缺测只与缺测相等**；缺测与有效值相遇时采用有效值；
  - 两个有效值不等即判定为**冲突，整次任务失败**，不产生任何结果文件。
- 读取、比对、写出全部按时间分块（`TIME_CHUNK`，默认 64 条）流式进行，
  内存占用与文件中的**总时次无关**，仅取决于分块大小与选区格点数。
- 结果先写入隐藏临时文件，完整写完并 `sync` 后原子 `rename` 发布；
  任何失败都清理临时文件，下载入口只对已发布文件开放。
- 任务状态存于 SQLite；**服务重启时所有未完成（QUEUED/RUNNING）任务
  一律标记为 FAILED**，已完成状态与完整文件保持一致。

## 目录结构

```
app/
  config.py     路径与分块参数（DATA_HOME、TIME_CHUNK 等）
  engine.py     校验、选区索引、分块流式读取、合并去重、原子写出
  jobs.py       SQLite 任务表、后台 worker、重启恢复
  web.py        Flask 路由：上传 / 提交 / 状态轮询 / 下载
  templates/    中文 Web 页面
scripts/
  make_samples.py  生成反向纬度、跨日期线、重叠冲突三类演示样例
tests/          unittest 端到端测试（13 项）
Dockerfile
docker-compose.yml
gunicorn.conf.py
```

## 一键启动（Docker Compose）

```bash
# 1) 构建并启动 Web 服务（监听 http://localhost:8000）
docker compose up --build -d

# 2) 生成演示数据到 ./samples/data
docker compose run --rm samples

# 3) 浏览器打开
#    http://localhost:8000
#    上传 samples/data 下三个 .nc 文件，表单已预填跨日期线选区
#    (west=170, east=-170, lat [-10,10])，提交后可下载单个 NetCDF。
```

演示样例：

| 文件 | 特征 |
| --- | --- |
| `day1_reversed_lat.nc` | 纬度轴**降序**存储，含一个缺测格点 |
| `day2_overlap_conflict.nc` | 与 day1 有重叠时刻；缺测可被填补，另有一个**故意冲突**点（lon 200 → -160） |
| `day3_clean.nc` | 无重叠的额外一天 |

- 用东界 `-170` 提交：选区不含 `-160` 冲突点，**成功**并下载结果；
- 把东界改成 `-160`：命中冲突，任务**整体失败**，页面显示冲突时刻与格点，
  且无任何残件可下载。

## 本地开发运行（无 Docker）

```bash
python -m pip install -r requirements.txt
export DATA_HOME=./.data
python -m scripts.make_samples ./samples/data
python -m gunicorn --config gunicorn.conf.py app.web:app
# 或开发模式：python -m app.web
```

运行测试：

```bash
python -m unittest discover -s tests -t . -v
```

## 关键设计说明

1. **格点对齐**：校验各文件排序后的纬度集合与经度数组完全一致；为每个
   文件建立独立的 lat/lon 存储位置映射，降序纬度在读取时即重排为升序。
2. **安全的轴重排**：netCDF4 不支持多维花式索引，先按一维时间索引读出
   分块，再用 `np.ix_` 开放网格在内存数组上独立重排 lat/lon，避免 NumPy
   交叉索引导致的格点错位。
3. **分块归并**：每个文件维护一个流，按时间戳做多路归并；每个时间戳只
   保留一个选区大小的累加器，逐文件合并并即时检测冲突。
4. **原子发布**：`TMP_DIR/.partial-<id>.nc` → `os.replace` 到
   `OUTPUT_DIR/subset_<id>.nc`；SUCCEEDED 状态仅在发布成功后写入。
5. **重启恢复**：`gunicorn` worker 初始化时调用 `recover_on_startup()`，
   将所有非终态任务置为 FAILED 并清理 `.partial-*` 临时文件。
6. **单进程多线程**：SQLite 与后台 worker 为进程内状态，因此 gunicorn
   固定 1 个 worker、多线程承载并发轮询。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATA_HOME` | `/data` | 上传、输出、临时文件与 SQLite 的根目录（Docker 卷） |
| `TIME_CHUNK` | `64` | 每次读取的时间记录数，控制内存上界 |
| `MAX_UPLOAD_MB` | `2048` | 单次请求上传大小上限（MB） |
| `PORT` | `8000` | 监听端口 |
