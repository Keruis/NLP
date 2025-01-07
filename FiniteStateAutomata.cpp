/*  通常情况下针对英语等印欧语系语言的词语切分任务
    可以采用基于有限状态自动机（Finite State Automata)
    融合正则表达式的方法完成
/*

有限状态自动机（Finite State Automaton, FSA） 是一个数学模型，用于表示具有有限状态的系统。它常被用于词语切分、词性标注、语法分析等自然语言处理任务中，尤其适合像英语这样的印欧语系语言的基础任务，比如词语切分。

1. 有限状态自动机的定义

有限状态自动机由以下五个部分组成：
	1.	一个有限的状态集合，表示系统可能处于的所有状态，用 Q 表示。
	2.	一个有限的输入符号集合，表示输入的字母表，例如字符或单词，用 Sigma 表示。
	3.	一个状态转移函数，定义了系统从一个状态接收某个输入后会转换到哪个状态，用 delta 表示，其形式为：delta(state, input) = next_state。
	4.	一个初始状态，即系统开始运行时所在的状态，用 q0 表示。
	5.	一组终止状态，表示系统完成匹配时的接受状态，用 F 表示。

2. 有限状态自动机的种类
	1.	确定性有限状态自动机（DFA）：
	•	每个状态对每个输入符号有且仅有一个转移，例如：delta(q1, a) = q2。
	•	设计更加高效，但对复杂语言规则需要更精细的设计。
	2.	非确定性有限状态自动机（NFA）：
	•	一个状态可以对同一个输入符号有多个可能的转移，例如：delta(q1, a) = {q2, q3}。
	•	更灵活，但计算能力与 DFA 相同，可以相互转换。

3. 在语言处理中的应用

有限状态自动机可以用来建模单词的拼写规则或语言的词法结构，特别是像英语这样的语言。

例子：英语单词切分

假设需要对句子中的单词 cats 进行切分：
	1.	构建有限状态自动机：
	•	初始状态是系统的开始状态，记为 q0。
	•	输入符号集合包括 c、a、t 和 s，记为 Sigma = {c, a, t, s}。
	•	状态转移规则如下：
	•	从初始状态 q0，接收字符 c 后，进入状态 q1，表示为：q0 -> c -> q1。
	•	接收字符 a 后，进入状态 q2，表示为：q1 -> a -> q2。
	•	接收字符 t 后，进入状态 q3，表示为：q2 -> t -> q3。
	•	接收字符 s 后，进入终止状态 q4，表示为：q3 -> s -> q4。
	•	终止状态表示输入字符串完成匹配，记为：F = {q4}。
	2.	运行过程：
输入字符串 cats，系统会按顺序接收每个字符，并根据状态转移规则依次匹配：
	•	初始状态 q0，接收 c 后进入状态 q1。
	•	状态 q1，接收 a 后进入状态 q2。
	•	状态 q2，接收 t 后进入状态 q3。
	•	状态 q3，接收 s 后进入终止状态 q4。
最终到达终止状态 q4，表明匹配成功。

基于有限状态自动机的优势
	•	高效性：有限状态自动机在匹配过程中只需线性扫描输入字符串，因此时间复杂度为线性级别（O(n)）。
	•	灵活性：可以通过设计状态和转移规则，捕获复杂的语言现象，如词缀、前缀和复合词等结构。

4. 扩展应用
	1.	正则表达式：有限状态自动机是正则表达式的理论基础，常用于模式匹配和词语切分。
	2.	词法分析器：编译器中的词法分析器使用有限状态自动机扫描源代码，将其分解为标记（tokens）。
	3.	分词任务：特别适合规则清晰的语言（如英语），通过有限状态自动机捕获语言中的词缀、复合词和语法规则。

*/
//!                                 FSA==FSM                               !\\

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

// 定义状态
enum State {
    q0, // 初始状态
    q1, // 接收字符 'c' 后的状态
    q2, // 接收字符 'a' 后的状态
    q3, // 接收字符 't' 后的状态
    q4, // 接收字符 's' 后的终止状态
    q5, // 额外的状态，演示非确定性
    ERROR // 错误状态
};

// 状态转移函数
unordered_set<State> transition(State current, char input) {
    unordered_set<State> nextStates;
    
    switch (current) {
        case q0:
            if (input == 'c') nextStates.insert(q1);
            if (input == 's') nextStates.insert(q5); // 演示空转移的非确定性
            break;
        case q1:
            if (input == 'a') nextStates.insert(q2);
            break;
        case q2:
            if (input == 't') nextStates.insert(q3);
            break;
        case q3:
            if (input == 's') nextStates.insert(q4); // 确保达到 q4
            break;
        case q5:
            if (input == 'a') nextStates.insert(q2); // 非确定性转移
            break;
        default:
            break;
    }

    return nextStates.empty() ? unordered_set<State>{ERROR} : nextStates;
}

// 非确定性自动机类
class NFA {
public:
    // 判断输入字符串是否符合规则
    bool isValidWord(const string& input) {
        unordered_set<State> currentStates = {q0}; // 初始状态为 q0
        
        for (char c : input) {
            unordered_set<State> nextStates;
            
            // 从当前状态集合中的每个状态，找到对应的下一个状态集合
            for (State state : currentStates) {
                unordered_set<State> result = transition(state, c);
                nextStates.insert(result.begin(), result.end());
            }

            currentStates = nextStates;
            
            // 如果没有合法的转移，说明进入了错误状态
            if (currentStates.count(ERROR)) return false;
        }

        // 只有到达终止状态 q4 才算匹配成功
        return currentStates.count(q4);
    }
};

// 主程序
int main() {
    string word;

    cout << "输入一个单词进行匹配（目标单词为 'cats'):";
    cin >> word;

    NFA nfa;
    if (nfa.isValidWord(word)) {
        cout << "匹配成功！输入的单词是 'cats'" << endl;
    } else {
        cout << "匹配失败！输入的单词不是 'cats'" << endl;
    }

    return 0;
}