"""
VBA Macro Generation for Enhanced Excel Reports
==============================================

Automated VBA macro generation for interactive Excel reports with advanced
features like data refresh, chart updates, and dynamic filtering.
"""

from typing import List, Dict, Any
from enum import Enum
from dataclasses import dataclass
import logging


class MacroType(Enum):
    """Types of VBA macros to generate"""

    DATA_REFRESH = "data_refresh"
    CHART_UPDATE = "chart_update"
    FILTER_CONTROL = "filter_control"
    DASHBOARD_NAV = "dashboard_navigation"
    EXPORT_CONTROL = "export_control"
    ANALYSIS_TOOLS = "analysis_tools"


@dataclass
class MacroConfig:
    """Configuration for VBA macro generation"""

    macro_type: MacroType
    target_worksheets: List[str]
    parameters: Dict[str, Any]
    auto_run: bool = False
    button_text: str = ""
    button_position: tuple[int, int] = (1, 1)


class VBAMacroGenerator:
    """Advanced VBA macro generator for Excel reports"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.macros: List[str] = []

    def generate_data_refresh_macro(self, config: MacroConfig) -> str:
        """Generate VBA macro for data refresh functionality"""

        macro_code = """
Sub RefreshSimulationData()
    '
    ' RefreshSimulationData Macro
    ' Automatically refresh all data connections and update charts
    '
    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual

    Dim ws As Worksheet
    Dim startTime As Double
    startTime = Timer

    ' Update status
    Range("A1").Value = "Refreshing data..."

    Try:
        ' Refresh all data connections
        For Each ws In ThisWorkbook.Worksheets
            If ws.Name <> "Dashboard" Then
                ws.Activate

                ' Refresh pivot tables if any
                Dim pt As PivotTable
                For Each pt In ws.PivotTables
                    pt.RefreshTable
                Next pt

                ' Refresh charts
                Dim cht As ChartObject
                For Each cht In ws.ChartObjects
                    cht.Chart.Refresh
                Next cht
            End If
        Next ws

        ' Update timestamp
        Worksheets("Dashboard").Range("B1").Value = "Last Updated: " & Now()

        ' Return to dashboard
        Worksheets("Dashboard").Activate

        ' Show completion message
        Dim duration As Double
        duration = Timer - startTime
        MsgBox "Data refresh completed in " & Round(duration, 2) & " seconds", vbInformation

    Catch:
        MsgBox "Error refreshing data: " & Err.Description, vbCritical

    Finally:
        Application.ScreenUpdating = True
        Application.Calculation = xlCalculationAutomatic
    End Try
End Sub
"""
        return macro_code

    def generate_chart_update_macro(self, config: MacroConfig) -> str:
        """Generate VBA macro for dynamic chart updates"""

        target_sheets = "', '".join(config.target_worksheets)

        macro_code = f'''
Sub UpdateDynamicCharts()
    '
    ' UpdateDynamicCharts Macro
    ' Update chart data ranges and formatting based on current data
    '
    Application.ScreenUpdating = False

    Dim targetSheets As Variant
    targetSheets = Array("{target_sheets}")

    Dim i As Integer
    Dim ws As Worksheet
    Dim cht As Chart
    Dim lastRow As Long

    For i = 0 To UBound(targetSheets)
        Set ws = Worksheets(targetSheets(i))

        ' Find last row with data
        lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row

        ' Update each chart in the worksheet
        Dim chartObj As ChartObject
        For Each chartObj In ws.ChartObjects
            Set cht = chartObj.Chart

            ' Update data range for time series charts
            If InStr(cht.ChartTitle.Text, "Time") > 0 Or InStr(cht.ChartTitle.Text, "Trend") > 0 Then
                UpdateTimeSeriesChart cht, ws, lastRow
            End If

            ' Update formatting
            UpdateChartFormatting cht
        Next chartObj
    Next i

    Application.ScreenUpdating = True
    MsgBox "Charts updated successfully", vbInformation
End Sub

Sub UpdateTimeSeriesChart(cht As Chart, ws As Worksheet, lastRow As Long)
    '
    ' Update time series chart data range
    '
    Dim newRange As Range
    Set newRange = ws.Range("A1").Resize(lastRow, 5)  ' Adjust columns as needed

    cht.SetSourceData Source:=newRange
End Sub

Sub UpdateChartFormatting(cht As Chart)
    '
    ' Apply consistent formatting to charts
    '
    With cht
        .ChartStyle = 15
        .HasTitle = True

        ' Format axes
        If .HasAxis(xlCategory) Then
            .Axes(xlCategory).TickLabels.Font.Size = 10
        End If

        If .HasAxis(xlValue) Then
            .Axes(xlValue).TickLabels.Font.Size = 10
        End If

        ' Format legend
        If .HasLegend Then
            .Legend.Font.Size = 10
            .Legend.Position = xlLegendPositionRight
        End If
    End With
End Sub
'''
        return macro_code

    def generate_filter_control_macro(self, config: MacroConfig) -> str:
        """Generate VBA macro for advanced filtering controls"""

        config.parameters.get("filter_columns", ["Colony", "Date", "Status"])

        macro_code = """
Sub CreateDynamicFilters()
    '
    ' CreateDynamicFilters Macro
    ' Create interactive filtering controls for data analysis
    '
    Dim ws As Worksheet
    Set ws = ActiveSheet

    ' Clear existing filters
    ws.AutoFilterMode = False

    ' Apply AutoFilter to data range
    Dim lastRow As Long
    Dim lastCol As Long
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row
    lastCol = ws.Cells(1, ws.Columns.Count).End(xlToLeft).Column

    Dim dataRange As Range
    Set dataRange = ws.Range(ws.Cells(1, 1), ws.Cells(lastRow, lastCol))
    dataRange.AutoFilter

    ' Create filter control buttons
    CreateFilterButtons ws

    MsgBox "Dynamic filters created successfully", vbInformation
End Sub

Sub CreateFilterButtons(ws As Worksheet)
    '
    ' Create custom filter control buttons
    '
    Dim btn As Button
    Dim btnRange As Range

    ' Filter by Colony button
    Set btnRange = ws.Range("A" & (ws.Cells(ws.Rows.Count, 1).End(xlUp).Row + 3))
    Set btn = ws.Buttons.Add(btnRange.Left, btnRange.Top, 100, 25)
    btn.Text = "Filter by Colony"
    btn.OnAction = "FilterByColony"

    ' Filter by Date Range button
    Set btnRange = ws.Range("C" & (ws.Cells(ws.Rows.Count, 1).End(xlUp).Row + 3))
    Set btn = ws.Buttons.Add(btnRange.Left, btnRange.Top, 100, 25)
    btn.Text = "Date Range Filter"
    btn.OnAction = "FilterByDateRange"

    ' Clear Filters button
    Set btnRange = ws.Range("E" & (ws.Cells(ws.Rows.Count, 1).End(xlUp).Row + 3))
    Set btn = ws.Buttons.Add(btnRange.Left, btnRange.Top, 100, 25)
    btn.Text = "Clear All Filters"
    btn.OnAction = "ClearAllFilters"
End Sub

Sub FilterByColony()
    '
    ' Interactive colony filtering
    '
    Dim userInput As String
    userInput = InputBox("Enter colony ID to filter (or 'All' for no filter):", "Colony Filter")

    If userInput <> "" And userInput <> "All" Then
        ActiveSheet.Range("A:A").AutoFilter Field:=1, Criteria1:="*" & userInput & "*"
    ElseIf userInput = "All" Then
        ActiveSheet.Range("A:A").AutoFilter Field:=1
    End If
End Sub

Sub FilterByDateRange()
    '
    ' Interactive date range filtering
    '
    Dim startDate As String
    Dim endDate As String

    startDate = InputBox("Enter start date (YYYY-MM-DD):", "Date Range Filter")
    If startDate = "" Then Exit Sub

    endDate = InputBox("Enter end date (YYYY-MM-DD):", "Date Range Filter")
    If endDate = "" Then Exit Sub

    ' Apply date filter (assuming date is in column B)
    ActiveSheet.Range("B:B").AutoFilter Field:=2, _
        Criteria1:=">=" & startDate, _
        Criteria2:="<=" & endDate, _
        Operator:=xlAnd
End Sub

Sub ClearAllFilters()
    '
    ' Clear all applied filters
    '
    If ActiveSheet.AutoFilterMode Then
        ActiveSheet.ShowAllData
    End If
    MsgBox "All filters cleared", vbInformation
End Sub
"""
        return macro_code

    def generate_dashboard_navigation_macro(self, config: MacroConfig) -> str:
        """Generate VBA macro for dashboard navigation"""

        worksheets = config.target_worksheets
        nav_buttons = "\n".join(
            [
                f'    CreateNavButton "{sheet}", {i + 1}, ws'
                for i, sheet in enumerate(worksheets)
            ]
        )

        macro_code = f"""
Sub CreateDashboardNavigation()
    '
    ' CreateDashboardNavigation Macro
    ' Create navigation buttons for easy worksheet switching
    '
    Dim ws As Worksheet
    Set ws = Worksheets("Dashboard")

    ' Clear existing navigation buttons
    ClearNavigationButtons ws

    ' Create navigation buttons
{nav_buttons}

    MsgBox "Navigation created successfully", vbInformation
End Sub

Sub CreateNavButton(sheetName As String, position As Integer, ws As Worksheet)
    '
    ' Create individual navigation button
    '
    Dim btn As Button
    Dim btnRange As Range

    ' Position buttons in a row
    Set btnRange = ws.Cells(20, position * 2)
    Set btn = ws.Buttons.Add(btnRange.Left, btnRange.Top, 120, 30)
    btn.Text = "Go to " & sheetName
    btn.OnAction = "NavigateToSheet"
    btn.Name = "Nav_" & sheetName
End Sub

Sub NavigateToSheet()
    '
    ' Navigate to selected worksheet
    '
    Dim buttonName As String
    Dim sheetName As String

    buttonName = Application.Caller
    sheetName = Replace(buttonName, "Nav_", "")

    If WorksheetExists(sheetName) Then
        Worksheets(sheetName).Activate
    Else
        MsgBox "Worksheet '" & sheetName & "' not found", vbExclamation
    End If
End Sub

Sub ClearNavigationButtons(ws As Worksheet)
    '
    ' Clear existing navigation buttons
    '
    Dim btn As Button
    For Each btn In ws.Buttons
        If InStr(btn.Name, "Nav_") > 0 Then
            btn.Delete
        End If
    Next btn
End Sub

Function WorksheetExists(sheetName As String) As Boolean
    '
    ' Check if worksheet exists
    '
    Dim ws As Worksheet
    On Error Resume Next
    Set ws = Worksheets(sheetName)
    WorksheetExists = Not ws Is Nothing
    On Error GoTo 0
End Function
"""
        return macro_code

    def generate_analysis_tools_macro(self, config: MacroConfig) -> str:
        """Generate VBA macro for advanced analysis tools"""

        macro_code = """
Sub CreateAnalysisTools()
    '
    ' CreateAnalysisTools Macro
    ' Create advanced analysis and calculation tools
    '
    Dim ws As Worksheet
    Set ws = Worksheets("Dashboard")

    ' Create analysis control panel
    CreateAnalysisControlPanel ws

    MsgBox "Analysis tools created successfully", vbInformation
End Sub

Sub CreateAnalysisControlPanel(ws As Worksheet)
    '
    ' Create analysis control panel with various tools
    '
    Dim startRow As Long
    startRow = 25

    ' Statistics Calculator button
    CreateAnalysisButton "Calculate Statistics", "CalculateStatistics", startRow, 1, ws

    ' Trend Analysis button
    CreateAnalysisButton "Trend Analysis", "PerformTrendAnalysis", startRow, 3, ws

    ' Correlation Analysis button
    CreateAnalysisButton "Correlation Analysis", "PerformCorrelationAnalysis", startRow, 5, ws

    ' Export Summary button
    CreateAnalysisButton "Export Summary", "ExportAnalysisSummary", startRow, 7, ws
End Sub

Sub CreateAnalysisButton(buttonText As String, macroName As String, row As Long, col As Long, ws As Worksheet)
    '
    ' Create individual analysis button
    '
    Dim btn As Button
    Dim btnRange As Range

    Set btnRange = ws.Cells(row, col)
    Set btn = ws.Buttons.Add(btnRange.Left, btnRange.Top, 140, 35)
    btn.Text = buttonText
    btn.OnAction = macroName
    btn.Name = "Analysis_" & Replace(buttonText, " ", "_")
End Sub

Sub CalculateStatistics()
    '
    ' Calculate comprehensive statistics for current data
    '
    Dim ws As Worksheet
    Set ws = ActiveSheet

    ' Find data range
    Dim lastRow As Long
    Dim lastCol As Long
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row
    lastCol = ws.Cells(1, ws.Columns.Count).End(xlToLeft).Column

    If lastRow <= 1 Then
        MsgBox "No data found for analysis", vbExclamation
        Exit Sub
    End If

    ' Create statistics summary
    Dim statsRange As Range
    Set statsRange = ws.Range(ws.Cells(lastRow + 3, 1), ws.Cells(lastRow + 10, 5))

    ' Clear existing statistics
    statsRange.Clear

    ' Add headers
    ws.Cells(lastRow + 3, 1).Value = "Statistical Summary"
    ws.Cells(lastRow + 3, 1).Font.Bold = True

    ' Calculate statistics for numeric columns
    Dim col As Long
    Dim statsRow As Long
    statsRow = lastRow + 5

    For col = 1 To lastCol
        If IsNumeric(ws.Cells(2, col).Value) Then
            ws.Cells(statsRow, 1).Value = ws.Cells(1, col).Value & " Statistics:"
            ws.Cells(statsRow, 1).Font.Bold = True

            ' Calculate mean, median, std dev
            Dim dataRange As Range
            Set dataRange = ws.Range(ws.Cells(2, col), ws.Cells(lastRow, col))

            ws.Cells(statsRow + 1, 1).Value = "Mean:"
            ws.Cells(statsRow + 1, 2).Value = Application.WorksheetFunction.Average(dataRange)

            ws.Cells(statsRow + 2, 1).Value = "Median:"
            ws.Cells(statsRow + 2, 2).Value = Application.WorksheetFunction.Median(dataRange)

            ws.Cells(statsRow + 3, 1).Value = "Std Dev:"
            ws.Cells(statsRow + 3, 2).Value = Application.WorksheetFunction.StDev(dataRange)

            statsRow = statsRow + 5
        End If
    Next col

    MsgBox "Statistics calculated successfully", vbInformation
End Sub

Sub PerformTrendAnalysis()
    '
    ' Perform trend analysis on time series data
    '
    Dim ws As Worksheet
    Set ws = ActiveSheet

    ' Implementation for trend analysis
    MsgBox "Trend analysis functionality - implementation pending", vbInformation
End Sub

Sub PerformCorrelationAnalysis()
    '
    ' Perform correlation analysis between variables
    '
    MsgBox "Correlation analysis functionality - implementation pending", vbInformation
End Sub

Sub ExportAnalysisSummary()
    '
    ' Export analysis summary to new workbook
    '
    Dim newWb As Workbook
    Set newWb = Workbooks.Add

    ' Copy dashboard and key worksheets
    ThisWorkbook.Worksheets("Dashboard").Copy Before:=newWb.Worksheets(1)

    ' Save as summary file
    Dim fileName As String
    fileName = "BSTEW_Analysis_Summary_" & Format(Now, "YYYY-MM-DD_HH-MM") & ".xlsx"

    newWb.SaveAs fileName
    MsgBox "Analysis summary exported as: " & fileName, vbInformation
End Sub
"""
        return macro_code

    def generate_all_macros(self, worksheet_names: List[str]) -> str:
        """Generate complete VBA module with all macros"""

        header = """
'
' BSTEW Simulation Analysis - VBA Macros
' =====================================
'
' Comprehensive VBA macro suite for interactive Excel reports
' Generated automatically by BSTEW Excel integration system
'
' Features:
' - Data refresh and update capabilities
' - Dynamic chart management
' - Interactive filtering controls
' - Dashboard navigation system
' - Advanced analysis tools
'

Option Explicit

"""

        # Generate all macro types
        configs = [
            MacroConfig(MacroType.DATA_REFRESH, worksheet_names, {}),
            MacroConfig(MacroType.CHART_UPDATE, worksheet_names, {}),
            MacroConfig(
                MacroType.FILTER_CONTROL,
                worksheet_names,
                {"filter_columns": ["Colony", "Date", "Status"]},
            ),
            MacroConfig(MacroType.DASHBOARD_NAV, worksheet_names, {}),
            MacroConfig(MacroType.ANALYSIS_TOOLS, worksheet_names, {}),
        ]

        all_macros = header

        for config in configs:
            if config.macro_type == MacroType.DATA_REFRESH:
                all_macros += self.generate_data_refresh_macro(config) + "\n\n"
            elif config.macro_type == MacroType.CHART_UPDATE:
                all_macros += self.generate_chart_update_macro(config) + "\n\n"
            elif config.macro_type == MacroType.FILTER_CONTROL:
                all_macros += self.generate_filter_control_macro(config) + "\n\n"
            elif config.macro_type == MacroType.DASHBOARD_NAV:
                all_macros += self.generate_dashboard_navigation_macro(config) + "\n\n"
            elif config.macro_type == MacroType.ANALYSIS_TOOLS:
                all_macros += self.generate_analysis_tools_macro(config) + "\n\n"

        return all_macros

    def save_macros_to_file(self, macros: str, file_path: str) -> None:
        """Save generated macros to VBA file"""

        try:
            with open(file_path, "w") as f:
                f.write(macros)
            self.logger.info(f"VBA macros saved to: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save VBA macros: {e}")
            raise


def create_macro_generator() -> VBAMacroGenerator:
    """Factory function to create VBA macro generator"""
    return VBAMacroGenerator()
