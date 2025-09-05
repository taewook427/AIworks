// test779 : ACH namer

package main

import (
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// go mod init example.com
// go mod tidy
// go build main.go

// may be detected as ransomware when process changes too many file names

// get names from folder
func getNames(path string) ([]string, error) {
	fs, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	res := make([]string, 0)
	for _, f := range fs {
		res = append(res, f.Name())
	}
	return res, nil
}

// split sentense into tokens, count frequency
func tokenize(name string) map[string]int {
	// normalize name
	idx := strings.LastIndex(name, ".")
	if idx != -1 {
		name = name[:idx]
	}
	name = strings.ToLower(name)
	sep := []string{"-", "_", ".", ",", "(", ")", "[", "]"}
	for _, s := range sep {
		name = strings.ReplaceAll(name, s, " ")
	}

	// remove last number
	count := 0
	for count < 4 && len(name) > 1 && '0' <= name[len(name)-1] && name[len(name)-1] <= '9' {
		count = count + 1
		name = name[:len(name)-1]
	}

	// split name, get frequency
	freq := make(map[string]int)
	for _, token := range strings.Split(name, " ") {
		if token != "" {
			freq[token] = freq[token] + 1
		}
	}
	return freq
}

// calculate ext-IDF values
func calculateIDF(names []string) map[string]float64 {
	documentCount := float64(len(names))
	docFreq := make(map[string]int)

	// calculate if token is included
	for _, name := range names {
		uniqueTokens := make(map[string]bool)
		for token := range tokenize(name) {
			uniqueTokens[token] = true
		}
		for token := range uniqueTokens {
			docFreq[token]++
		}
	}

	// ext-IDF = log( A * (all+1) / (in+1) + B * (in+1) )
	idf := make(map[string]float64)
	for term, freq := range docFreq {
		idf[term] = math.Log(WEIGHT_IDF*(documentCount+1)/(float64(freq)+1) + WEIGHT_SF*(float64(freq)+1))
	}
	return idf
}

// calculate cosine similarity
func cosineSimilarity(vec1 []float64, vec2 []float64) float64 {
	var dotProduct float64
	var normA, normB float64
	for i := 0; i < len(vec1); i++ {
		dotProduct += vec1[i] * vec2[i]
		normA += vec1[i] * vec1[i]
		normB += vec2[i] * vec2[i]
	}
	if normA == 0 || normB == 0 {
		return 0.0
	}
	return math.Exp(WEIGHT_COS * math.Log(0.000001+dotProduct/(math.Sqrt(normA)*math.Sqrt(normB))))
}

// make TF-IDF matrix of entire names
func buildMatrix(fileNames []string) ([][]float64, []string) {
	// calculate IDF, prepare word map, TF data
	idf := calculateIDF(fileNames)
	allTerms := make(map[string]bool)
	tfMaps := make([]map[string]int, len(fileNames))
	var wg sync.WaitGroup

	// tokenize and register word
	div := len(fileNames) / THREAD
	localMap := make([]map[string]bool, THREAD+1)
	for i := 0; i < THREAD; i++ {
		wg.Add(1)
		go func(num int) {
			defer wg.Done()
			dict := make(map[string]bool)
			for j := num * div; j < (num+1)*div; j++ {
				tfMaps[j] = tokenize(fileNames[j])
				for term := range tfMaps[j] {
					dict[term] = true
				}
			}
			localMap[num] = dict
		}(i)
	}
	dict := make(map[string]bool)
	for j := THREAD * div; j < len(fileNames); j++ {
		tfMaps[j] = tokenize(fileNames[j])
		for term := range tfMaps[j] {
			dict[term] = true
		}
	}
	localMap[THREAD] = dict
	wg.Wait()
	for _, tfMap := range localMap {
		for term := range tfMap {
			allTerms[term] = true
		}
	}

	// make sorted word dictionary
	var vocab []string
	for term := range allTerms {
		vocab = append(vocab, term)
	}
	slices.Sort(vocab)

	// calculate TF-IDF
	matrix := make([][]float64, len(fileNames))
	for i := 0; i < THREAD; i++ {
		wg.Add(1)
		go func(num int) {
			defer wg.Done()
			for j := num * div; j < (num+1)*div; j++ {
				matrix[j] = make([]float64, len(vocab))
				for k, term := range vocab {
					matrix[j][k] = float64(tfMaps[j][term]) * idf[term]
				}
			}
		}(i)
	}
	for j := THREAD * div; j < len(fileNames); j++ {
		for j := THREAD * div; j < len(fileNames); j++ {
			matrix[j] = make([]float64, len(vocab))
			for k, term := range vocab {
				matrix[j][k] = float64(tfMaps[j][term]) * idf[term]
			}
		}
	}
	wg.Wait()
	return matrix, vocab
}

// HierarchicalClustering repeat
func HierarchicalClustering(vectors [][]float64) []MergeStep {
	// init cluster info, steps
	numVec := len(vectors)
	if numVec <= 1 {
		return nil
	}
	clusters := make(map[int][]int)
	for i := 0; i < numVec; i++ {
		clusters[i] = []int{i}
	}
	mergeSteps := make([]MergeStep, 0)

	// repeat until all clusters merged
	for len(clusters) > 1 {
		// worker pool channels
		taskChan := make(chan [2]int)
		resultChan := make(chan MergeResult, 32)
		var wg sync.WaitGroup
		wg.Add(THREAD)

		// start workers
		for i := 0; i < THREAD; i++ {
			go func() {
				defer wg.Done()
				for pair := range taskChan {
					// calculate distance between cluster
					key1, key2 := pair[0], pair[1]
					dist := math.Inf(1)
					for _, idx1 := range clusters[key1] {
						for _, idx2 := range clusters[key2] {
							currentDist := 1.0 - cosineSimilarity(vectors[idx1], vectors[idx2])
							if currentDist < dist {
								dist = currentDist
							}
						}
					}
					resultChan <- MergeResult{From: key1, To: key2, Distance: dist}
				}
			}()
		}

		// push tasks
		go func() {
			clusterKeys := make([]int, 0, len(clusters))
			for k := range clusters {
				clusterKeys = append(clusterKeys, k)
			}
			sort.Ints(clusterKeys)
			for i := 0; i < len(clusterKeys); i++ {
				for j := i + 1; j < len(clusterKeys); j++ {
					taskChan <- [2]int{clusterKeys[i], clusterKeys[j]}
				}
			}
			close(taskChan)
			wg.Wait()
			close(resultChan)
		}()

		// get result, find minimal dist
		var minDist = math.Inf(1)
		var mergeFrom, mergeTo int
		for res := range resultChan {
			if res.Distance < minDist {
				minDist = res.Distance
				mergeFrom = res.From
				mergeTo = res.To
			}
		}

		// merge and update
		newCluster := append(clusters[mergeFrom], clusters[mergeTo]...)
		newClusterKey := len(mergeSteps) + numVec
		clusters[newClusterKey] = newCluster
		delete(clusters, mergeFrom)
		delete(clusters, mergeTo)
		mergeSteps = append(mergeSteps, MergeStep{
			Left:       mergeFrom,
			Right:      mergeTo,
			Distance:   minDist,
			NewCluster: newClusterKey,
		})
	}
	return mergeSteps
}

// make tree result
func printTree(f *os.File, node int, mergeSteps []MergeStep, fileNames []string, indent string) {
	// leaf node (original file)
	if node < len(fileNames) {
		fmt.Fprintf(f, "%s|-%s\n", indent, fileNames[node])
		return
	}

	// merged cluster, find merging position
	var step MergeStep
	for _, s := range mergeSteps {
		if s.NewCluster == node {
			step = s
			break
		}
	}
	fmt.Fprintf(f, "%s|-[%.4f]\n", indent, step.Distance)

	// recursive call
	printTree(f, step.Left, mergeSteps, fileNames, indent+"  ")
	printTree(f, step.Right, mergeSteps, fileNames, indent+"  ")
}

// cut merging by threshold, get final groups
func getFinalClusters(mergeSteps []MergeStep, numVec int) [][]int {
	// init with all independent elements
	clusters := make(map[int]bool)
	for i := 0; i < numVec; i++ {
		clusters[i] = true
	}

	// merge, update
	for _, step := range mergeSteps {
		if step.Distance > THRESHOLD {
			break
		}
		clusters[step.NewCluster] = true
		delete(clusters, step.Left)
		delete(clusters, step.Right)
	}

	// find result
	var result [][]int
	for clusterKey := range clusters {
		result = append(result, getMembers(mergeSteps, clusterKey, numVec))
	}
	return result
}

// find elements from cluster key
func getMembers(mergeSteps []MergeStep, key int, numVec int) []int {
	if key < numVec {
		return []int{key}
	}
	for _, step := range mergeSteps {
		if step.NewCluster == key {
			leftMembers := getMembers(mergeSteps, step.Left, numVec)
			rightMembers := getMembers(mergeSteps, step.Right, numVec)
			return append(leftMembers, rightMembers...)
		}
	}
	return nil
}

var THREAD int = 128
var WEIGHT_IDF float64 = 0.01
var WEIGHT_SF float64 = 1.0
var WEIGHT_COS float64 = 2.5
var THRESHOLD float64 = 0.2

var tgtDir string = ""
var isReal bool = false
var isRename bool = false
var renameVec []string = make([]string, 0)

type MergeResult struct {
	From     int
	To       int
	Distance float64
}

type MergeStep struct {
	Left       int
	Right      int
	Distance   float64
	NewCluster int
}

func main() {
	var err error
	for i, arg := range os.Args {
		switch arg {
		case "-tgt":
			tgtDir = strings.ReplaceAll(os.Args[i+1], "\\", "/")
			if tgtDir[len(tgtDir)-1] != '/' {
				tgtDir = tgtDir + "/"
			}
		case "-real":
			isReal = true
		case "-rename":
			isRename = true
		case "-idf":
			WEIGHT_IDF, err = strconv.ParseFloat(os.Args[i+1], 64)
			if err != nil {
				fmt.Println(err)
			}
		case "-sf":
			WEIGHT_SF, err = strconv.ParseFloat(os.Args[i+1], 64)
			if err != nil {
				fmt.Println(err)
			}
		case "-cos":
			WEIGHT_COS, err = strconv.ParseFloat(os.Args[i+1], 64)
			if err != nil {
				fmt.Println(err)
			}
		case "-th":
			THRESHOLD, err = strconv.ParseFloat(os.Args[i+1], 64)
			if err != nil {
				fmt.Println(err)
			}
		case "-i":
			renameVec = append(renameVec, os.Args[i+1])
		}
	}

	if isRename {
		// simple rename mode
		pool, _ := getNames(tgtDir)
		fmt.Printf("%d names from %s\n", len(pool), tgtDir)
		fmt.Println("replace vector")
		for i, r := range renameVec {
			fmt.Printf("%d: %s\n", i, r)
		}

		// start real rename
		if isReal {
			for _, name := range pool {
				for num, frag := range renameVec {
					if strings.Contains(name, frag) {
						err = os.Rename(tgtDir+name, fmt.Sprintf("%ss%03d %s", tgtDir, num, name))
						if err != nil {
							fmt.Println(err)
						}
						break
					}
				}
			}
		}

	} else {
		// auto split mode
		pool, _ := getNames(tgtDir)
		fmt.Printf("%d names from %s\n", len(pool), tgtDir)
		vec, wordDict := buildMatrix(pool)
		fmt.Printf("vec dimension %d\n", len(wordDict))
		steps := HierarchicalClustering(vec)
		fmt.Printf("%d steps of merging\n", len(steps))

		// write words list
		f0, _ := os.Create("words.txt")
		defer f0.Close()
		for _, word := range wordDict {
			fmt.Fprintln(f0, word)
		}

		// tree view of merging
		f1, _ := os.Create("tree.txt")
		defer f1.Close()
		node := steps[len(steps)-1]
		printTree(f1, node.NewCluster, steps, pool, "")

		// split result by threshold
		f2, _ := os.Create("groups.txt")
		defer f2.Close()
		cluster := getFinalClusters(steps, len(pool))
		for i, group := range cluster {
			fmt.Fprintf(f2, "[%d]: %d\n", i, len(group))
			for _, idx := range group {
				fmt.Fprintln(f2, "  "+pool[idx])
			}
		}

		// start real split
		if isReal {
			fmt.Printf("moving files at %s\n", tgtDir)
			for i, group := range cluster {
				name := fmt.Sprintf("%sgroup_%d_%d/", tgtDir, i, len(group))
				err = os.Mkdir(name, os.ModePerm)
				if err != nil {
					fmt.Println(err)
				}
				for _, idx := range group {
					err = os.Rename(tgtDir+pool[idx], name+pool[idx])
					if err != nil {
						fmt.Println(err)
					}
				}
			}
		}
	}
}

// main -tgt . -rename -i Q1 -i Q2 -i Q3
// main -tgt . -real -idf 0 -sf 1.0 -cos 3.0 -th 0.5

