package main

import (
	"reflect"
	"testing"

	formula "github.com/DTrader-store/formula-go"
)

func TestCompiledProgramMatchesSourceSessionAcrossSymbols(t *testing.T) {
	engine := formula.NewFormulaEngine()
	source := `XG: CROSS(CLOSE, MA(CLOSE, 2));`
	program, err := engine.Compile(source)
	if err != nil {
		t.Fatal(err)
	}
	requests := []request{
		{Formula: source, Symbol: "600000.SH", Bars: []bar{
			{Date: "2024-01-02", Close: 10},
			{Date: "2024-01-03", Close: 9},
			{Date: "2024-01-04", Close: 11},
			{Date: "2024-01-05", Close: 9},
			{Date: "2024-01-06", Close: 12},
		}},
		{Formula: source, Symbol: "000001.SZ", Bars: []bar{
			{Date: "2024-01-02", Close: 8},
			{Date: "2024-01-03", Close: 10},
			{Date: "2024-01-04", Close: 9},
			{Date: "2024-01-05", Close: 11},
			{Date: "2024-01-06", Close: 8},
		}},
	}

	for _, req := range requests {
		got := evaluate(req, engine, program)
		want := evaluateFromSource(t, req)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("compiled evaluation differs for %s: got %#v, want %#v", req.Symbol, got, want)
		}
	}
}

func TestLogicalContinuationMatchesSingleLineFormula(t *testing.T) {
	engine := formula.NewFormulaEngine()
	multiline := "XG: CLOSE > OPEN\nAND CLOSE > 0\nOR OPEN < 0;"
	program, err := engine.Compile(normalizeFormula(multiline))
	if err != nil {
		t.Fatal(err)
	}
	reference, err := engine.Compile("XG: CLOSE > OPEN AND CLOSE > 0 OR OPEN < 0;")
	if err != nil {
		t.Fatal(err)
	}
	req := request{Formula: multiline, Symbol: "600000.SH", Bars: []bar{
		{Date: "2024-01-02", Open: 10, Close: 11},
		{Date: "2024-01-03", Open: 11, Close: 10},
	}}
	got := evaluate(req, engine, program)
	want := evaluate(req, engine, reference)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("multiline logical operators changed result: got %#v, want %#v", got, want)
	}
}

func TestUserFormulaWithLeadingAndLinesParses(t *testing.T) {
	source := `A1:=REF(MA(C,5),29);
A2:=REF(MA(C,5),21);
A3:=REF(MA(C,5),13);
A4:=REF(MA(C,5),5);

S1:=A2/A1-1;
S2:=A3/A2-1;
S3:=A4/A3-1;

JIASU:=S1>-0.005 AND S1<0.05
AND S2>S1+0.006
AND S2>0.008
AND S3>S2+0.006
AND S3<0.09;

ZHANG:=A4/A1>1.05
AND A4/A1<1.30;

PINGHUA:=REF(COUNT(MA(C,5)>REF(MA(C,5),1),24),5)>=16;

XIAOYANG:=REF(COUNT(ABS(C/REF(C,1)-1)>0.06,25),5)<=2;

BOXH:=HHV(H,5);
BOXL:=LLV(L,5);
TOP0:=REF(HHV(H,20),5);

PINTAI:=BOXH/BOXL<1.07
AND ABS(C/REF(C,4)-1)<0.03;

GAOWEI:=MA(C,5)>TOP0*0.95
AND MA(C,5)<TOP0*1.04;

WEIPO:=C/REF(C,1)<1.045;

XG:BARSCOUNT(C)>60
AND JIASU
AND ZHANG
AND PINGHUA
AND XIAOYANG
AND PINTAI
AND GAOWEI
AND WEIPO;`
	if _, err := formula.NewFormulaEngine().Compile(normalizeFormula(source)); err != nil {
		t.Fatalf("user formula failed after normalizing logical continuations: %v", err)
	}
}

func evaluateFromSource(t *testing.T, req request) response {
	t.Helper()
	engine := formula.NewFormulaEngine()
	session, err := engine.NewStreamingSessionFromSource(req.Formula, nil)
	if err != nil {
		t.Fatal(err)
	}
	result := response{Symbol: req.Symbol, Signals: make([]signal, 0)}
	for _, item := range req.Bars {
		current, err := session.Append(formula.NewMarketData(item.Open, item.Close, item.High, item.Low, item.Volume, item.Amount))
		if err != nil {
			t.Fatal(err)
		}
		value, ok := selectedValue(current)
		if ok && value != 0 {
			result.Signals = append(result.Signals, signal{Date: item.Date, Value: value})
		}
	}
	return result
}
