REM The checking half of `office bench`, run by LibreOffice itself.
REM
REM Three entry points, each driven by a job file the kernel writes, one
REM record per task:
REM
REM   View      out, sheet, workbook        -> the sheet's used area as text
REM   Formula   out, sheet, workbook, cell  -> the formula kept there
REM   Check     out, sheet, range, formula, then input/answer pairs
REM                                           -> PASS or FAIL per pair
REM   Probe     out, sheet, range, input, then candidate formulas
REM                                           -> what each computes there
REM
REM Check is the benchmark's own rule: the model's formula goes into the
REM first cell of the answer range and is filled across the rest, the way a
REM person drags a formula; the workbook is recalculated; every answer cell
REM must match the answer workbook. Numbers within a relative 1e-6, text
REM exactly once trimmed, an error never.
REM
REM Both end by terminating LibreOffice: a headless instance that has run a
REM macro otherwise stays up, and the kernel waits on it to exit.

Option Explicit

REM Where the kernel leaves job files. The entry points take no argument:
REM a fixed place costs nothing and cannot be misquoted.
REM
REM **One launch does a whole batch.** A LibreOffice start in the guest
REM costs most of a minute, and launching per task made the benchmark a
REM day long. Each line of a job is one task's record, fields separated by
REM tabs, and each record writes its own output file -- so a workbook that
REM will not open costs its own task and not the batch.
Const JOBS = "/tmp/office/bench/"

Sub View
	Batch(JOBS & "view.job", 0)
End Sub

Sub Formula
	Batch(JOBS & "formula.job", 1)
End Sub

Sub Check
	Batch(JOBS & "check.job", 2)
End Sub

Sub Probe
	Batch(JOBS & "probe.job", 3)
End Sub

Sub Cells
	Batch(JOBS & "cells.job", 4)
End Sub

Sub Answer
	Batch(JOBS & "answer.job", 5)
End Sub

Sub Batch(job As String, kind As Integer)
	Dim sfa As Object, inp As Object, l As String
	REM UTF-8 both ways, through the stream services. Basic's own Open and
	REM Print use the system's legacy encoding, which turned a minus sign, an
	REM en dash and a middle dot into '?' in every cell read and every answer
	REM written -- right answers compared wrong, and conditions on values that
	REM were not in the sheet.
	sfa = createUnoService("com.sun.star.ucb.SimpleFileAccess")
	inp = createUnoService("com.sun.star.io.TextInputStream")
	inp.setInputStream(sfa.openFileRead(ConvertToURL(job)))
	inp.setEncoding("UTF-8")
	Do While Not inp.isEOF()
		l = inp.readLine()
		If Len(l) > 0 Then Record(Split(l, Chr(9)), kind)
	Loop
	inp.closeInput()
	StarDesktop.terminate()
End Sub

REM One record, its failure kept to itself.
Sub Record(r As Variant, kind As Integer)
	On Error GoTo bad
	Select Case kind
		Case 0
			WriteText(r(0), ViewOf(r(1), r(2)))
		Case 1
			WriteText(r(0), FormulaOf(r(1), r(2), r(3)))
		Case 2
			Dim t As String, i As Integer
			t = ""
			For i = 4 To UBound(r) - 1 Step 2
				t = t & OneCase(r(i), r(i + 1), r(1), r(2), r(3)) & Chr(10)
			Next i
			WriteText(r(0), t)
		Case 3
			WriteText(r(0), Probed(r))
		Case 4
			WriteText(r(0), CellsOf(r(1), r(2)))
		Case 5
			WriteText(r(0), Answered(r))
	End Select
	Exit Sub
bad:
	WriteText(r(0), "ERROR " & Err & " " & Error$)
End Sub

Function Hidden() As Variant
	Dim a(0) As New com.sun.star.beans.PropertyValue
	a(0).Name = "Hidden"
	a(0).Value = True
	Hidden = a()
End Function

Function OpenBook(path As String) As Object
	OpenBook = StarDesktop.loadComponentFromURL(ConvertToURL(path), "_blank", 0, Hidden())
End Function

REM A workbook to change, opened as an untitled copy.
REM
REM **The guest reads the benchmark off a read-only disk**, so LibreOffice
REM opened every input as a read-only document -- and on one, a single cell's
REM formula still went in while copyRange and setArrayFormula did nothing at
REM all, with no error: H3 right, H4 empty, the array range empty. On the
REM host the same files were writable and everything worked. AsTemplate gives
REM an editable copy whatever the file's own rights are.
Function OpenEditable(path As String) As Object
	Dim a(1) As New com.sun.star.beans.PropertyValue
	a(0).Name = "Hidden"
	a(0).Value = True
	a(1).Name = "AsTemplate"
	a(1).Value = True
	OpenEditable = StarDesktop.loadComponentFromURL(ConvertToURL(path), "_blank", 0, a())
End Function

REM The named sheet, or the one the workbook was saved showing.
Function SheetOf(doc As Object, name As String) As Object
	If name <> "" Then
		If doc.Sheets.hasByName(name) Then
			SheetOf = doc.Sheets.getByName(name)
			Exit Function
		End If
	End If
	On Error Resume Next
	Dim s As Object
	s = doc.CurrentController.ActiveSheet
	If IsNull(s) Or IsEmpty(s) Then s = doc.Sheets.getByIndex(0)
	SheetOf = s
End Function

Function ReadLines(job As String) As Variant
	Dim f As Integer, l As String, n As Integer
	Dim out(200) As String
	f = FreeFile
	Open job For Input As #f
	n = 0
	Do While Not EOF(f) And n < 200
		Line Input #f, l
		out(n) = l
		n = n + 1
	Loop
	Close #f
	ReDim Preserve out(n - 1) As String
	ReadLines = out()
End Function

Sub WriteText(path As String, text As String)
	Dim sfa As Object, o As Object, url As String
	sfa = createUnoService("com.sun.star.ucb.SimpleFileAccess")
	url = ConvertToURL(path)
	If sfa.exists(url) Then sfa.kill(url)
	o = createUnoService("com.sun.star.io.TextOutputStream")
	o.setOutputStream(sfa.openFileWrite(url))
	o.setEncoding("UTF-8")
	o.writeString(text & Chr(10))
	o.closeOutput()
End Sub

Function ColName(c As Long) As String
	Dim s As String, n As Long
	n = c + 1
	s = ""
	Do While n > 0
		s = Chr(65 + ((n - 1) Mod 26)) & s
		n = (n - 1) \ 26
	Loop
	ColName = s
End Function

Function ViewOf(sheet As String, book As String) As String
	Dim doc As Object, s As Object, cur As Object
	Dim r As Long, c As Long, lastR As Long, lastC As Long, t As String, ln As String, cell As Object, v As String
	doc = OpenBook(book)
	s = SheetOf(doc, sheet)
	cur = s.createCursor()
	cur.gotoEndOfUsedArea(False)
	lastR = cur.RangeAddress.EndRow
	lastC = cur.RangeAddress.EndColumn
	If lastR > 59 Then lastR = 59
	If lastC > 15 Then lastC = 15
	t = "Sheet " & s.Name & ", used area A1:" & ColName(cur.RangeAddress.EndColumn) & (cur.RangeAddress.EndRow + 1) & Chr(10)
	For r = 0 To lastR
		ln = ""
		For c = 0 To lastC
			cell = s.getCellByPosition(c, r)
			v = cell.String
			If Len(v) > 30 Then v = Left(v, 30) & "..."
			If cell.Type = com.sun.star.table.CellContentType.FORMULA Then v = v & " {" & cell.Formula & "}"
			If v <> "" Then
				If ln <> "" Then ln = ln & " | "
				ln = ln & ColName(c) & (r + 1) & "=" & v
			End If
		Next c
		If ln <> "" Then t = t & ln & Chr(10)
	Next r
	doc.close(True)
	ViewOf = t
End Function

REM The whole used area, a row a line and a tab between cells, nothing cut:
REM View is for a reader and stops at sixty rows and thirty characters, which
REM loses exactly the value a question names when it is long or far down.
Function CellsOf(sheet As String, book As String) As String
	Dim doc As Object, s As Object, cur As Object, r As Long, c As Long, lastR As Long, lastC As Long
	Dim t As String, ln As String
	doc = OpenBook(book)
	s = SheetOf(doc, sheet)
	cur = s.createCursor()
	cur.gotoEndOfUsedArea(False)
	lastR = cur.RangeAddress.EndRow
	lastC = cur.RangeAddress.EndColumn
	If lastR > 999 Then lastR = 999
	If lastC > 39 Then lastC = 39
	t = ""
	For r = 0 To lastR
		ln = ""
		For c = 0 To lastC
			If c > 0 Then ln = ln & Chr(9)
			ln = ln & Replace(Replace(s.getCellByPosition(c, r).String, Chr(9), " "), Chr(10), " ")
		Next c
		t = t & ln & Chr(10)
	Next r
	doc.close(True)
	CellsOf = t
End Function

REM A copy of a workbook with the answer on a sheet of its own, for a person
REM to open: the question as a bold title, then a row a result -- its label,
REM and the live formula that computes it from the data sheet, so the answer is
REM the spreadsheet's own arithmetic and stays so if the data changes. The new
REM sheet is first and active, which ODF keeps with the document: beside the
REM data it was off screen in any window narrower than the table. Saved as ODF
REM to r(2); the original is opened as a template and not touched.
REM   out, input, output, sheet name, title, then label, formula pairs
Function Answered(r As Variant) As String
	Dim d As Object, s As Object, i As Integer, row As Long, c As Object
	Dim a(0) As New com.sun.star.beans.PropertyValue
	d = OpenEditable(r(1))
	If d.Sheets.hasByName(r(3)) Then d.Sheets.removeByName(r(3))
	d.Sheets.insertNewByName(r(3), 0)
	s = d.Sheets.getByName(r(3))
	row = 2
	For i = 5 To UBound(r) - 1 Step 2
		c = s.getCellByPosition(0, row)
		c.String = r(i)
		c.CharWeight = com.sun.star.awt.FontWeight.SEMIBOLD
		c = s.getCellByPosition(1, row)
		If Left(r(i + 1), 2) = "{=" Then
			c.setArrayFormula(ApiSyntax(Mid(r(i + 1), 2, Len(r(i + 1)) - 2)))
		Else
			c.FormulaLocal = r(i + 1)
		End If
		c.CellBackColor = RGB(255, 246, 228)
		row = row + 1
	Next i
	d.calculateAll()
	s.Columns.getByIndex(0).OptimalWidth = True
	s.Columns.getByIndex(1).OptimalWidth = True
	REM The title after the widths, so it runs on over the empty cells rather
	REM than sizing column A to the whole sentence.
	c = s.getCellByPosition(0, 0)
	c.String = r(4)
	c.CharWeight = com.sun.star.awt.FontWeight.BOLD
	c.CharHeight = 14
	c.CellBackColor = RGB(255, 236, 200)
	d.CurrentController.setActiveSheet(s)
	a(0).Name = "FilterName"
	a(0).Value = "calc8"
	d.storeToURL(ConvertToURL(r(2)), a())
	d.close(True)
	Answered = "OK"
End Function

REM The formula a workbook keeps in one cell, as a person would type it:
REM the reference solution, for the oracle arm.
Function FormulaOf(sheet As String, book As String, cell As String) As String
	Dim doc As Object, s As Object, c As Object
	doc = OpenBook(book)
	s = SheetOf(doc, sheet)
	c = s.getCellRangeByName(cell)
	FormulaOf = ""
	If c.Type = com.sun.star.table.CellContentType.FORMULA Then FormulaOf = c.FormulaLocal
	doc.close(True)
End Function

Function Same(a As Object, g As Object) As Boolean
	Same = False
	REM An error is an answer too: #N/A where the answer has #N/A is right.
	If a.getError() <> 0 Then
		Same = (g.getError() <> 0 And a.String = g.String)
		Exit Function
	End If
	If g.getError() <> 0 Then Exit Function
	Dim numeric As Boolean
	numeric = (a.Type = com.sun.star.table.CellContentType.VALUE)
	If a.Type = com.sun.star.table.CellContentType.FORMULA Then
		numeric = (a.FormulaResultType2 = com.sun.star.sheet.FormulaResult.VALUE)
	End If
	If a.Type = com.sun.star.table.CellContentType.EMPTY Then
		Same = (Trim(g.String) = "" Or g.Value = 0)
		Exit Function
	End If
	If numeric Then
		If g.Type = com.sun.star.table.CellContentType.TEXT Then Exit Function
		Dim tol As Double
		tol = Abs(a.Value) * 1E-6
		If tol < 1E-6 Then tol = 1E-6
		Same = (Abs(a.Value - g.Value) <= tol)
	Else
		Same = (Trim(a.String) = Trim(g.String))
	End If
End Function

REM Commas to semicolons, outside quoted text.
Function ApiSyntax(f As String) As String
	Dim i As Long, c As String, q As Boolean, o As String
	q = False
	o = ""
	For i = 1 To Len(f)
		c = Mid(f, i, 1)
		If c = Chr(34) Then q = Not q
		If c = "," And Not q Then c = ";"
		o = o & c
	Next i
	ApiSyntax = o
End Function

REM One test case. An array formula, {=...}, has two readings and both are
REM tried: one array per cell, copied down the range the way a person copies a
REM single-cell array formula, or one array over the whole range. Accepting
REM either is the generous reading of a formula written without saying which.
Function OneCase(inp As String, ans As String, sheet As String, rng As String, formula As String) As String
	If Left(formula, 2) = "{=" And Right(formula, 1) = "}" Then
		Dim first As String
		first = Attempt(inp, ans, sheet, rng, formula, 1)
		If first = "PASS" Then
			OneCase = first
			Exit Function
		End If
		Dim whole As String
		whole = Attempt(inp, ans, sheet, rng, formula, 2)
		If whole = "PASS" Then OneCase = whole Else OneCase = first
	Else
		OneCase = Attempt(inp, ans, sheet, rng, formula, 0)
	End If
End Function

REM Put a formula over a range the way Attempt's mode says.
Sub Fill(s As Object, r As Object, formula As String, mode As Integer)
	Dim addr As Object, first As Object, x As Long, y As Long, body As String
	addr = r.RangeAddress
	first = s.getCellByPosition(addr.StartColumn, addr.StartRow)
	If mode = 0 Then
		REM Copied cell by cell rather than fillAuto: a copy adjusts relative
		REM references exactly as a drag does, and fillAuto filled nothing
		REM in the guest -- H4 held no formula at all -- where the host's
		REM filled every cell.
		first.FormulaLocal = formula
		For y = addr.StartRow To addr.EndRow
			For x = addr.StartColumn To addr.EndColumn
				If x <> addr.StartColumn Or y <> addr.StartRow Then
					s.copyRange(s.getCellByPosition(x, y).CellAddress, first.RangeAddress)
				End If
			Next x
		Next y
	Else
		REM setArrayFormula takes the API's syntax, whose separator is the
		REM semicolon, so commas outside quoted text become semicolons.
		body = ApiSyntax(Mid(formula, 2, Len(formula) - 2))
		If mode = 2 Then
			r.setArrayFormula(body)
		Else
			first.setArrayFormula(body)
			For y = addr.StartRow To addr.EndRow
				For x = addr.StartColumn To addr.EndColumn
					If x <> addr.StartColumn Or y <> addr.StartRow Then
						s.copyRange(s.getCellByPosition(x, y).CellAddress, first.RangeAddress)
					End If
				Next x
			Next y
		End If
	End If
End Sub

REM What each candidate computes over the answer range of one input workbook,
REM a line each: "ERR <cell> <text>" at the first error, else "OK" and the
REM cells' values. No answer workbook is opened: this is what a person could
REM see by trying the formula, and it is all a choice between candidates may
REM rest on. The workbook is opened once and the range put back between
REM candidates, so the next one starts from what the first saw.
Function Probed(r As Variant) As String
	On Error GoTo bad
	Dim d As Object, s As Object, rg As Object, saved As Variant, t As String, i As Integer
	d = OpenEditable(r(3))
	s = SheetOf(d, r(1))
	rg = s.getCellRangeByName(r(2))
	saved = rg.getFormulaArray()
	t = ""
	For i = 4 To UBound(r)
		t = t & ProbeOne(d, s, rg, r(i)) & Chr(10)
		rg.setFormulaArray(saved)
	Next i
	d.close(True)
	Probed = t
	Exit Function
bad:
	Probed = t & "ERR probe " & Err & " " & Error$ & Chr(10)
End Function

REM One candidate, its failure kept to itself: a formula Calc will not take
REM raises here, and that is an answer about the formula rather than a reason
REM to lose the candidates after it.
Function ProbeOne(d As Object, s As Object, rg As Object, f As String) As String
	On Error GoTo bad
	Dim addr As Object, x As Long, y As Long, c As Object, ln As String
	addr = rg.RangeAddress
	If Left(f, 2) = "{=" And Right(f, 1) = "}" Then
		Fill(s, rg, f, 1)
	Else
		Fill(s, rg, f, 0)
	End If
	d.calculateAll()
	ln = "OK"
	For y = addr.StartRow To addr.EndRow
		For x = addr.StartColumn To addr.EndColumn
			c = s.getCellByPosition(x, y)
			If c.getError() <> 0 Then
				ln = "ERR " & ColName(x) & (y + 1) & " " & c.String
				x = addr.EndColumn
				y = addr.EndRow
			Else
				ln = ln & Chr(9) & Left(c.String, 60)
			End If
		Next x
	Next y
	ProbeOne = ln
	Exit Function
bad:
	ProbeOne = "ERR fill " & Err
End Function

REM mode 0: the formula in the first cell, filled across the range.
REM mode 1: an array formula in the first cell, copied to every other cell.
REM mode 2: one array formula over the whole range.
Function Attempt(inp As String, ans As String, sheet As String, rng As String, formula As String, mode As Integer) As String
	Dim d As Object, a As Object, s As Object, sa As Object, r As Object, addr As Object
	Dim first As Object, row As Object, x As Long, y As Long, ga As Object, gg As Object, body As String
	d = OpenEditable(inp)
	s = SheetOf(d, sheet)
	r = s.getCellRangeByName(rng)
	addr = r.RangeAddress
	Fill(s, r, formula, mode)
	d.calculateAll()
	a = OpenBook(ans)
	If sheet <> "" Then
		sa = SheetOf(a, sheet)
	ElseIf a.Sheets.hasByName(s.Name) Then
		sa = a.Sheets.getByName(s.Name)
	Else
		sa = SheetOf(a, "")
	End If
	Attempt = "PASS"
	For y = addr.StartRow To addr.EndRow
		For x = addr.StartColumn To addr.EndColumn
			ga = sa.getCellByPosition(x, y)
			gg = s.getCellByPosition(x, y)
			If Not Same(ga, gg) Then
				Attempt = "FAIL " & ColName(x) & (y + 1) & " got [" & Left(gg.String, 40) & "] want [" & Left(ga.String, 40) & "] holding [" & Left(gg.Formula, 60) & "]"
				x = addr.EndColumn
				y = addr.EndRow
			End If
		Next x
	Next y
	d.close(True)
	a.close(True)
End Function


