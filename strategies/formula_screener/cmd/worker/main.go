package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"

	formula "github.com/DTrader-store/formula-go"
)

type bar struct {
	Date   string  `json:"date"`
	Open   float64 `json:"open"`
	High   float64 `json:"high"`
	Low    float64 `json:"low"`
	Close  float64 `json:"close"`
	Volume float64 `json:"vol"`
	Amount float64 `json:"amount"`
}

type request struct {
	Formula string `json:"formula"`
	Symbol  string `json:"symbol"`
	Bars    []bar  `json:"bars"`
}

type signal struct {
	Date  string  `json:"date"`
	Value float64 `json:"value"`
}

type response struct {
	Symbol  string   `json:"symbol"`
	Signals []signal `json:"signals,omitempty"`
	Error   string   `json:"error,omitempty"`
}

var logicalContinuation = regexp.MustCompile(`(?im)\r?\n[ \t]*(AND|OR)\b`)

func normalizeFormula(source string) string {
	return logicalContinuation.ReplaceAllString(source, " $1")
}

func main() {
	input := bufio.NewReaderSize(os.Stdin, 64*1024)
	output := bufio.NewWriter(os.Stdout)
	defer output.Flush()
	engine := formula.NewFormulaEngine()
	var compiledFormula string
	var compiledProgram *formula.Program
	var compileError error
	hasCompiled := false
	for {
		line, err := input.ReadBytes('\n')
		if err != nil && err != io.EOF {
			fmt.Fprintln(os.Stderr, err)
			return
		}
		if len(line) > 0 {
			var req request
			if decodeErr := json.Unmarshal(line, &req); decodeErr != nil {
				write(output, response{Error: decodeErr.Error()})
			} else {
				if !hasCompiled || req.Formula != compiledFormula {
					compiledFormula = req.Formula
					compiledProgram, compileError = engine.Compile(normalizeFormula(req.Formula))
					hasCompiled = true
				}
				if compileError != nil {
					write(output, response{Error: compileError.Error()})
				} else {
					write(output, evaluate(req, engine, compiledProgram))
				}
			}
		}
		if err == io.EOF {
			return
		}
	}
}

func evaluate(req request, engine *formula.FormulaEngine, program *formula.Program) response {
	result := response{Symbol: req.Symbol, Signals: make([]signal, 0)}
	session, err := engine.NewStreamingSession(program, nil)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	markets := make([]*formula.MarketData, len(req.Bars))
	for i, item := range req.Bars {
		markets[i] = formula.NewMarketData(item.Open, item.Close, item.High, item.Low, item.Volume, item.Amount)
	}
	currentResults, appendErr := session.AppendMany(markets)
	if appendErr != nil {
		result.Error = appendErr.Error()
		result.Signals = nil
		return result
	}
	for i, current := range currentResults {
		item := req.Bars[i]
		value, ok := selectedValue(current)
		if ok && value != 0 && !math.IsNaN(value) && !math.IsInf(value, 0) {
			result.Signals = append(result.Signals, signal{Date: item.Date, Value: value})
		}
	}
	return result
}

func selectedValue(result *formula.StreamingResult) (float64, bool) {
	if result == nil {
		return 0, false
	}
	for _, output := range result.Outputs {
		if output != nil && (output.Name == "XG" || output.Name == "选股" || output.Name == "SELECT" || output.Name == "选股条件") {
			return output.Value, true
		}
	}
	for _, name := range []string{"XG", "选股", "SELECT", "选股条件"} {
		if value, ok := result.Variables[name]; ok {
			return value, true
		}
	}
	for i := len(result.Outputs) - 1; i >= 0; i-- {
		if result.Outputs[i] != nil {
			return result.Outputs[i].Value, true
		}
	}
	return 0, false
}

func write(writer *bufio.Writer, value response) {
	encoded, err := json.Marshal(value)
	if err != nil {
		encoded = []byte(fmt.Sprintf(`{"error":%q}`, err.Error()))
	}
	_, _ = writer.Write(append(encoded, '\n'))
	_ = writer.Flush()
}
