#include <cstring>
#include <cctype>
#include <iostream>
#include "token.h"

string source;
vector<Token> tokens;

int line, length, pos;
Token get_name();
Token get_num();
Token get_word();

string convert_bk(const char* text) {
    vector<pair<char*, string>> dictionary = {
        {const_cast<char*>("\xEC\x95\x88\xEB\x85\x95\x20\xEC\x96\x98\xEB\x93\xA4\xEC\x95\x84"), "#start"},
        {const_cast<char*>("\xEC\x98\xA4\xEB\x8A\x98\x20\xEC\x88\x98\xEC\x97\x85\x20\xEC\x97\xAC\xEA\xB8\xB0\xEA\xB9\x8C\xEC\xA7\x80"), "#exit"},
        {const_cast<char*>("\xEC\x97\x90\xED\x97\xA4\xEC\x9D\xB4"), "/*"},
        {const_cast<char*>("\xEC\x97\xAC\xEB\x9F\xAC\xEB\xB6\x84\x20\xEC\x9D\xB4\xEB\x9F\xAC\xEB\xA9\xB4\x20\xEC\x95\x88\xEB\x90\xA9\xEB\x8B\x88\xEB\x8B\xA4"), "*/"},

        {const_cast<char*>("\xED\x91\x9C\xED\x98\x84\xEB\xA0\xA5\x21"), "write"},
        {const_cast<char*>("\xED\x92\x80\xEC\x96\xB4\xEB\xB3\xB4\xEC\x84\xB8\xEC\x9A\x94\x21"), "writeln"},
        {const_cast<char*>("\xEA\xB4\x80\xEC\x8B\xAC\xEB\xB0\x9B\xEA\xB3\xA0\x20\xEC\x8B\xB6\xEC\x96\xB4\x3F"), "read"},
        {const_cast<char*>("\xEC\x9E\x90\x20\xEC\x95\x89\xEC\x95\x84\xEB\xB3\xB4\xEC\x84\xB8\xEC\x9A\x94"), "push"},
        {const_cast<char*>("\xEB\x84\x88\x20\xEC\x95\x84\xEC\x9B\x83\xEC\x9D\xB4\xEC\x95\xBC"), "pop"},
        {const_cast<char*>("\xEB\x94\xB0\xEB\x9D\xBC\xEC\x98\xA4\xEB\x8B\x88\x7E"), "set"},

        {const_cast<char*>("\xEC\x97\xAC\xEB\x9F\xAC\xEB\xB6\x84\x20\xEB\xA7\x9E\xEC\x8A\xB5\xEB\x8B\x88\xEA\xB9\x8C\x3F"), "if"},
        {const_cast<char*>("\xEC\x95\x84\xEB\x8B\x8C\xEA\xB0\x80\x3F"), "else"},
        {const_cast<char*>("\xEA\xB3\xB5\xEB\xB6\x80\xEC\x95\x88\xED\x95\xB4\x3F"), "while"},
        {const_cast<char*>("\xEB\xB6\x88\xEA\xB4\x91\xEB\x8F\x99"), "end"},

        {const_cast<char*>("\xEB\x8D\x94\xED\x95\xB4"), "+"},
        {const_cast<char*>("\xEB\xB9\xBC"), "-"},
        {const_cast<char*>("\xEA\xB3\xB1\xED\x95\xB4"), "*"},
        {const_cast<char*>("\xEB\x82\x98\xEB\x88\xA0"), "/"},
        {const_cast<char*>("\xEC\x82\xBC\xEB\xB6\x84\xEC\x9D\x98\x20\xEC\x9D\xBC\x20\xEB\xB2\x95\xEC\xB9\x99"), "%"},
        {const_cast<char*>("\xEA\xB0\x99\xEC\x95\x84"), "=="},
        {const_cast<char*>("\xEB\x8B\xAC\xEB\x9D\xBC"), "!="},
        {const_cast<char*>("\xEC\xBB\xA4"), ">"},
        {const_cast<char*>("\xEC\x9E\x91\xEC\x95\x84"), "<"},

        {const_cast<char*>("\xEC\xA0\x90"), "i"},
        {const_cast<char*>("\xEC\x9B\x94\x20\xEB\xAA\xA8\xEC\x9D\x98\xEA\xB3\xA0\xEC\x82\xAC"), "p"},
        {const_cast<char*>("\xEC\x9B\x94\x20\xED\x95\x99\xEB\xA0\xA5\xED\x8F\x89\xEA\xB0\x80"), "n"},
        {const_cast<char*>("\xEC\xA3\xBC\xEC\x98\xA5\xEA\xB0\x99\xEC\x9D\x80\x20\xEA\xB8\xB0\xEC\xB6\x9C\xEB\xAC\xB8\xEC\xA0\x9C"), "1p"},
        {const_cast<char*>("\xEA\xB5\x90\xEA\xB3\xBC\xEC\x84\x9C\x20\xEC\x88\x98\xEC\xA4\x80\x20\xEB\xAC\xB8\xEC\xA0\x9C"), "1n"},

        {const_cast<char*>("\x62\x6B"), "A"},
        {const_cast<char*>("\x6C\x6F\x76\x65"), "B"},
        {const_cast<char*>("\xEA\xB9\x80"), "C"},
        {const_cast<char*>("\xEB\xB3\x91"), "D"},
        {const_cast<char*>("\xEA\xB4\x80"), "E"},
    };

    string result;
    size_t len = strlen(text);
    vector<char> buffer(text, text + len);

    size_t pos = 0;
    while (pos < buffer.size()) {
        bool matched = false;
        for (const auto& pair : dictionary) {
            const char* target = pair.first;
            int sz = strlen(target);
            if (strncmp(&buffer[pos], target, sz) == 0) {
                result += pair.second;
                pos += sz;
                matched = true;
                break;
            }
        }
        if (!matched) {
            result += buffer[pos];
            ++pos;
        }
    }

    return result;
}

void get_tokens() {
    line = 1;
    length = (int)source.length();
    pos = 0;
    bool iscomment = false;
    tokens.clear();
    while (pos < length) {
        if (source[pos] == '\n') line++;
        if (isspace(source[pos])) {
            pos++;
            continue;
        }
        switch (source[pos])
        {
        case '.':
            tokens.push_back(Token(Dot, ".", line));
            pos++;
            break;
        case ',':
            tokens.push_back(Token(Comma, ",", line));
            pos++;
            break;
        case '+':
            tokens.push_back(Token(Plus, "+", line));
            pos++;
            break;
        case '%':
            tokens.push_back(Token(Remain, "%", line));
            pos++;
            break;
        case '>':
            tokens.push_back(Token(Grt, ">", line));
            pos++;
            break;
        case '<':
            tokens.push_back(Token(Sml, "<", line));
            pos++;
            break;
        case '*':
            if (source[pos + 1] == '/') {
                tokens.push_back(Token(C_ed, "*/", line));
                iscomment = false;
                pos++;
            }
            else {
                tokens.push_back(Token(Mult, "*", line));
            }
            pos++;
            break;
        case '/':
            if (source[pos + 1] == '*') {
                tokens.push_back(Token(C_st, "/*", line));
                iscomment = true;
                pos++;
            }
            else {
                tokens.push_back(Token(Divi, "/", line));
            }
            pos++;
            break;
        case '=':
            pos++;
            if (source[pos] == '=') {
                tokens.push_back(Token(Eql, "==", line));
            }
            else {
                cout << "invalid char at line " << line << " is ignored : =" << endl;
            }
            pos++;
            break;
        case '!':
            pos++;
            if (source[pos] == '=') {
                tokens.push_back(Token(Neq, "!=", line));
            }
            else {
                cout << "invalid char at line " << line << " is ignored : !" << endl;
            }
            pos++;
            break;
        case '-':
            if (isdigit(source[pos + 1])) {
                tokens.push_back(get_num());
            }
            else {
                tokens.push_back(Token(Minus, "-", line));
                pos++;
            }
            break;
        default:
            if (islower(source[pos]) || source[pos] == '#') {
                tokens.push_back(get_word());
            }
            else if (isdigit(source[pos])) {
                tokens.push_back(get_num());
            }
            else if ('A' <= source[pos] && source[pos] <= 'E') {
                tokens.push_back(get_name());
            }
            else if (!iscomment) {
                cout << "invalid char at line " << line << " is ignored : " << source[pos] << endl;
                pos++;
            }
            else {
                pos++;
            }
        }
    }
}

Token get_name() {
    int temp = pos + 1;
    while (temp < length && 'A' <= source[temp] && source[temp] <= 'E') temp++;
    Token out = Token(T_id, source.substr(pos, temp-pos), line);
    pos = temp;
    return out;
}

Token get_num() {
    int temp = pos + 1;
    bool isminus = (source[pos] == '-');
    while (temp < length && isdigit(source[temp])) temp++;
    Token out = Token(Other, "", line);
    if (source[temp] == 'i') {
        out = Token(T_literal, source.substr(pos, temp-pos), line);
    } else if (source[temp] == 'p') {
        if (isminus) {
            tokens.push_back(Token(Minus, "-", line));
            out = Token(T_literal, source.substr(pos + 1, temp - pos - 1), line);
        }
        else {
            out = Token(T_literal, source.substr(pos, temp - pos), line);
        }
    } else if (source[temp] == 'n') {
        if (isminus) {
            tokens.push_back(Token(Minus, "-", line));
            out = Token(T_literal, "-" + source.substr(pos + 1, temp - pos - 1), line);
        }
        else {
            out = Token(T_literal, "-" + source.substr(pos, temp - pos), line);
        }
    }
    else {
        cout << "invalid char at line " << line << " is ignored : " << source[pos] << endl;
    }
    pos = temp + 1;
    return out;
}

Token get_word() {
    int temp = pos + 1;
    Token out = Token(Other, "", line);
    while (temp < length && islower(source[temp])) temp++;
    string cmp = source.substr(pos, temp-pos);
    if (cmp == "if") {
        out = Token(B_if, "if", line);
    }
    else if (cmp == "else") {
        out = Token(B_else, "else", line);
    }
    else if (cmp == "while") {
        out = Token(B_while, "while", line);
    }
    else if (cmp == "end") {
        out = Token(B_ed, "end", line);
    }
    else if (cmp == "push") {
        out = Token(Push, "push", line);
    }
    else if (cmp == "pop") {
        out = Token(Pop, "pop", line);
    }
    else if (cmp == "write") {
        out = Token(Write, "write", line);
    }
    else if (cmp == "writeln") {
        out = Token(Writeln, "writeln", line);
    }
    else if (cmp == "read") {
        out = Token(Read, "read", line);
    }
    else if (cmp == "set") {
        out = Token(Set, "set", line);
    }
    else if (cmp == "#start") {
        out = Token(P_st, "#start", line);
    }
    else if (cmp == "#exit") {
        out = Token(P_ed, "#exit", line);
    }
    else {
        cout << "invalid keyword at line " << line << " : " << cmp << endl;
    }
    pos = temp;
    return out;
}
