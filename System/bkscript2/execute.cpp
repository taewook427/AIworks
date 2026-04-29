#include <vector>
#include <iostream>
#include "compile.h"
#include "execute.h"

class IntStack {
private:
    vector<int> data;

public:
    void clear() {
        data.clear();
    }

    void push(int value) {
        data.push_back(value);
    }

    int pop() {
        int temp = 0;
        if (!data.empty()) {
            temp = data.back();
            data.pop_back();
        }
        else {
            cerr << "Error : Stack Underflow" << endl;
        }
        return temp;
    }

    int get(int index) {
        if (index < data.size()) {
            return data[index];
        }
        else {
            cerr << "Error : Stack Index Out" << endl;
            return 0;
        }
    }

    void set(int index, int d) {
        if (index < data.size()) {
            data[index] = d;
        }
        else {
            cerr << "Error : Stack Index Out" << endl;
        }
    }
};

int pc;
IntStack stack, addmem;

void execute() {
	pc = 0;
	stack.clear();
	addmem.clear();
    int pgrsz = codes.size(), op0 = 0, op1 = 0;
    while (pc < pgrsz) {
        switch (codes[pc].kind)
        {
        case ADD:
            op1 = stack.pop();
            op0 = stack.pop();
            stack.push(op0 + op1);
            break;
        case SUB:
            op1 = stack.pop();
            op0 = stack.pop();
            stack.push(op0 - op1);
            break;
        case MUL:
            op1 = stack.pop();
            op0 = stack.pop();
            stack.push(op0 * op1);
            break;
        case DIV:
            op1 = stack.pop();
            op0 = stack.pop();
            stack.push(op0 / op1);
            break;
        case REM:
            op1 = stack.pop();
            op0 = stack.pop();
            stack.push(op0 % op1);
            break;
        case EQL:
            op1 = stack.pop();
            op0 = stack.pop();
            if (op0 == op1) stack.push(1);
            else stack.push(0);
            break;
        case NEQ:
            op1 = stack.pop();
            op0 = stack.pop();
            if (op0 == op1) stack.push(0);
            else stack.push(1);
            break;
        case GRT:
            op1 = stack.pop();
            op0 = stack.pop();
            if (op0 > op1) stack.push(1);
            else stack.push(0);
            break;
        case SML:
            op1 = stack.pop();
            op0 = stack.pop();
            if (op0 < op1) stack.push(1);
            else stack.push(0);
            break;
        case PUSH:
            stack.push(stack.get(codes[pc].data));
            break;
        case POP:
            stack.set(codes[pc].data, stack.pop());
            break;
        case JMP:
            pc = pc + codes[pc].data - 1;
            break;
        case JMPF:
            if (stack.pop() == 0) pc = pc + codes[pc].data - 1;
            break;
        case WRT:
            cout << stack.pop() << " ";
            break;
        case WRL:
            cout << stack.pop() << endl;
            break;
        case RED:
            op0 = 0;
            cout << ">> ";
            cin >> op0;
            stack.set(codes[pc].data, op0);
            break;
        case MPUSH:
            addmem.push(stack.pop());
            break;
        case MPOP:
            stack.set(codes[pc].data, addmem.pop());
            break;
        case CPUSH:
            stack.push(codes[pc].data);
            break;
        case INIT:
            stack.clear();
            for (op0 = 0; op0 < codes[pc].data; op0++) stack.push(0);
            break;
        default:
            cerr << "Error : Unknown OpCode" << endl;
        }
        pc++;
    }
}