# Laptop Data Dictionary

## Overview
This document defines the schema, data types, and engineering logic applied to the cleaned laptop dataset (`clean_laptop_data.csv`). The dataset contains cleaned physical hardware parameters, extracted performance metrics, and pricing data suitable for analytics and predictive modeling.

---

## Field Specifications

| Column Name | Data Type | Units / Format | Description | Example Values |
| :--- | :--- | :--- | :--- | :--- |
| **`Company`** | `string` | Categorical | Brand or manufacturer of the laptop. | `Apple`, `Dell`, `HP`, `Asus` |
| **`TypeName`** | `string` | Categorical | Physical form factor / target user classification. | `Ultrabook`, `Notebook`, `Gaming`, `Workstation` |
| **`Inches`** | `float` | Inches (`in`) | Physical diagonal screen size. Imputed using group medians if missing. | `13.3`, `15.6`, `17.3` |
| **`Ram`** | `integer` | Gigabytes (`GB`) | System Random Access Memory (RAM) capacity. | `4`, `8`, `16`, `32` |
| **`Weight`** | `float` | Kilograms (`kg`) | System mass weight. | `1.37`, `2.10`, `0.99` |
| **`OpSys`** | `string` | Categorical | Consolidated Operating System family. | `Windows`, `Mac`, `Linux`, `Chrome OS`, `Other / No OS` |
| **`Touchscreen`** | `integer` | Binary (`0` / `1`) | Flag indicating if display supports touch interaction. | `0`, `1` |
| **`IPS`** | `integer` | Binary (`0` / `1`) | Flag indicating if display uses In-Plane Switching panel tech. | `0`, `1` |
| **`PPI`** | `float` | Pixels Per Inch | Calculated pixel density derived from resolution dimensions and screen size. | `141.21`, `226.98` |
| **`Clock_Speed_GHz`** | `float` | Gigahertz (`GHz`) | Base/boost CPU frequency extracted from raw processor string. | `1.8`, `2.5`, `3.1` |
| **`Cpu_Brand`** | `string` | Categorical | Primary processor manufacturer and broad performance family. | `Intel Core i7`, `Intel Core i5`, `AMD`, `Other Intel` |
| **`Gpu_Brand`** | `string` | Categorical | Graphics processor manufacturer. | `Nvidia`, `Intel`, `AMD` |
| **`Gpu_Type`** | `string` | Categorical | Hardware architecture classification (`Dedicated` graphics card vs. `Integrated` CPU graphics). | `Integrated`, `Dedicated` |
| **`Gpu_Tier`** | `string` | Categorical | Performance tier bucket grouping GPU models by market segment. | `Nvidia High Gaming`, `Intel HD/UHD (Standard Integrated)`, `Nvidia Quadro (Workstation)` |
| **`SSD_GB`** | `integer` | Gigabytes (`GB`) | Solid State Drive storage capacity. Parsed from compound drive strings. | `0`, `128`, `256`, `512` |
| **`HDD_GB`** | `integer` | Gigabytes (`GB`) | Hard Disk Drive storage capacity. | `0`, `500`, `1024`, `2048` |
| **`Flash_GB`** | `integer` | Gigabytes (`GB`) | eMMC or Flash storage capacity. | `0`, `32`, `64` |
| **`Hybrid_GB`** | `integer` | Gigabytes (`GB`) | SSHD (Hybrid Solid State Hard Drive) capacity. | `0`, `1000` |
| **`Price`** | `float` | Numeric Currency | Target variable representing laptop retail price. | `1339.69`, `789.00` |

---

## Key Transformations Applied

1. **Missing Data Imputation:** Single missing entries in `Inches` imputed using median value of corresponding `TypeName`.
2. **Text Standardization:** Removed trailing text indicators (`GB`, `kg`) and standard whitespace across all rows.
3. **Regex Feature Extraction:** 
   * `ScreenResolution` converted into `Touchscreen`, `IPS`, and calculated screen sharpness (`PPI`).
   * `Cpu` decomposed into numeric `Clock_Speed_GHz` and `Cpu_Brand`.
   * `Memory` split across four dedicated numeric drive capacity features (`SSD_GB`, `HDD_GB`, `Flash_GB`, `Hybrid_GB`).
   * `Gpu` normalized to handle Unicode encoding errors (`<U+039C>`) and missing space formatting before extracting brand, type, and performance tiers.
4. **Redundancy Cleanup:** Original unstructured compound text columns (`Cpu`, `Gpu`, `ScreenResolution`, `Memory`) removed from final export.
