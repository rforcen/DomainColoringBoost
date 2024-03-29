#ifndef zCompilerH
#define zCompilerH

#include <complex>
#include "zSymbols.h"

using namespace std;
typedef u_char byte;


class zCompiler {
public:
    typedef complex<float>ComplexFloat;
    
    vector<string>fname = {
            "SIN", "COS", "TAN", "EXP", "LOG", "LOG10", "INT", "SQRT", "ASIN",
            "ACOS", "ATAN", "ABS", "C", "PI", "PHI"};
    int lfname;
    
    constexpr static const float PHI=1.6180339887;
    static const int maxCode=16000, maxStack=150;
    
    string s; // expression to evaluate;
    int ixs;
    
    int sym; // actual sym
    char ch; // actual ch
    float nval; // actual numerical value
    
    string ident; // actual id,

    bool err;
    
    string errMessage;
    
    // compiler
    int32_t code[maxCode]; // assume sizeof(int32_t)==sizeof(float)
    int pc, codeSize;

public:
    zCompiler() : lfname((int)fname.size()) {
        static_assert(sizeof(int32_t)==sizeof(float), "sizeof(int32_t)!=sizeof(float)");
    }
    
    string getErrorMessage() {
        return errMessage;
    }
    string toUpper(string strToConvert)  {
        std::transform(strToConvert.begin(), strToConvert.end(), strToConvert.begin(), ::toupper);
        return strToConvert;
    }
    string toLower(string strToConvert)  {
        std::transform(strToConvert.begin(), strToConvert.end(), strToConvert.begin(), ::tolower);
        return strToConvert;
    }
    
    bool compile(string expr) {
        pc = ixs = 0;
        ident = "";
        s = expr;
        getch();
        err = false;
        getsym();
        Ce0();
        
        Gen(END);
        codeSize = pc;
        return err;
    }
    
    ComplexFloat execute(ComplexFloat z) {
        ComplexFloat stack[maxStack]; // local stack to support concurrency
        ComplexFloat CI = ComplexFloat(0., 1.);
        
        int pc=0, sp=0;

        for (bool end=false; !end; ) {
            switch (code[pc]) {
                case PUSHC:
                    pc++;
                    stack[sp] = ComplexFloat(*(float*)(code+pc), 0.);
                    sp++;
                    pc++;
                    break;
                case PUSHZ:
                    pc++;
                    stack[sp] = z;
                    sp++;
                    break;
                case PUSHI:
                    pc++;
                    stack[sp] = CI;
                    sp++;
                    break;
                case PLUS:
                    sp--;
                    stack[sp - 1] += stack[sp];
                    pc++;
                    break;
                case MINUS:
                    sp--;
                    stack[sp - 1] -= stack[sp];
                    pc++;
                    break;
                case MULT:
                    sp--;
                    stack[sp - 1] *= stack[sp];
                    pc++;
                    break;
                case DIV:
                    sp--;
                    stack[sp - 1] /= stack[sp];
                    pc++;
                    break;
                case POWER:
                    sp--;
                    stack[sp - 1] = pow(stack[sp - 1], stack[sp]);
                    pc++;
                    break;
                case NEG:
                    stack[sp - 1] = -stack[sp - 1];
                    pc++;
                    break;
                    
                case FSIN:
                    stack[sp - 1] = sin(stack[sp - 1]);
                    pc++;
                    break;
                case FCOS:
                    stack[sp - 1] = cos(stack[sp - 1]);
                    pc++;
                    break;
                case FTAN:
                    stack[sp - 1] = tan(stack[sp - 1]);
                    pc++;
                    break;
                case FASIN:
                    stack[sp - 1] = asin(stack[sp - 1]);
                    pc++;
                    break;
                case FACOS:
                    stack[sp - 1] = acos(stack[sp - 1]);
                    pc++;
                    break;
                case FATAN:
                    stack[sp - 1] = atan(stack[sp - 1]);
                    pc++;
                    break;
                case FEXP:
                    stack[sp - 1] = exp(stack[sp - 1]);
                    pc++;
                    break;
                case FLOG:
                    stack[sp - 1] = log(stack[sp - 1]);
                    pc++;
                    break;
                case FLOG10:
                    stack[sp - 1] = log10(stack[sp - 1]);
                    pc++;
                    break;
                case FSQRT:
                    stack[sp - 1] = sqrt(stack[sp - 1]);
                    pc++;
                    break;
                case FABS:
                    stack[sp - 1] = abs(stack[sp - 1]);
                    pc++;
                    break;
                case FINT:
                    stack[sp - 1] = ComplexFloat((int)stack[sp - 1].real(), (int)stack[sp - 1].imag());
                    pc++;
                    break;
                case FC:
                    sp--;
                    pc++;
                    stack[sp - 1] = ComplexFloat(stack[sp - 1].real(), stack[sp].real());
                    break;
                case END:
                    end=true; break;
                default:
                    err = true;
                    end = true;
                    break;
            }
        }
        if (sp != 0)
            return stack[sp - 1];
        else
            return ComplexFloat(0, 0);
    }
    
private:
    bool islower(char c) {        return (c >= 'a' && c <= 'z');    }
    bool isupper(char c) {        return (c >= 'A' && c <= 'Z');    }
    bool isalpha(char c) {        return (islower(c) || isupper(c));}
    bool isdigit(char c) {        return (c >= '0' && c <= '9');    }
    bool isalnum(char c) {        return (isalpha(c) || isdigit(c));}
    
    // get next char from *s
    char getch() {
        ch = 0;
        if (ixs < (int)s.length()) {
            ch = s[ixs];
            ixs++;
        }
        return ch;
    }
    
    void ungetch() {
        ixs--;
    }
    
    // get next symbol
    int getsym() {
        int i;
        
        sym = SNULL;
        ident = "";
        
        // skip blanks
        while (ch != 0 && ch <= ' ')
            getch();
        // detect symbol
        if (isalpha(ch)) { // ident
            ident = "";
            for (i = 0; isalnum(ch) || ch == '_'; i++) {
                ident += ch;
                getch();
            }
            sym = IDENT_i;
            ident = toUpper(ident); // look up for 'x' or 't'
            
            if (ident == "Z")
                sym = IDENT_z;
            else if (ident == "I")
                sym = IDENT_i;
            else {
                // is a funct ?
                for (i = 0; i < lfname; i++) {
                    if (ident == fname[i]) {
                        sym = i + FSIN; // first symbol offset
                        break;
                    }
                }
                if (i >= lfname) {
                    sym = 0;
                    error("unknown symbol:" + ident);
                }
            }
        }
        else {
            if (isdigit(ch)) { // number (float) take care of dddd.ddde-dd
                for (i = 0; isdigit(ch) || ch == '.' || ch == 'e' ||
                     ch == 'E'; i++) {
                    ident += ch;
                    getch();
                }
                sym = NUMBER;
                nval = stof(ident);
            }
            else {
                switch (ch) {
                    case '+':
                        sym = PLUS;
                        break;
                    case '-':
                        sym = MINUS;
                        break;
                    case '*':
                        sym = MULT;
                        break;
                    case '/':
                        sym = DIV;
                        break;
                    case '(':
                        sym = OPAREN;
                        break;
                    case ')':
                        sym = CPAREN;
                        break;
                    case '^':
                        sym = POWER;
                        break;
                    case ',':
                        sym = PERIOD;
                        break;
                    case 0:
                        sym = SNULL;
                        break;
                        
                    default:
                        sym = SNULL;
//                        error("character not recognized: " + ch);
                        break;
                }
                getch();
            }
        }
        return sym;
    }
    
    void error(string text) {
        errMessage = text;
        err = true;
    }
    
    void Gen(int token, float f) { // code Generation
        code[pc++] = token;
        *(float*)(code + pc) = f; // as sizeof(float) == sizeof(int)
        pc++;
    }
    
    void Gen(int token, int i) {
        code[pc++] = token;
        code[pc++] = i;
    }
    
    void Gen(int token) {
        code[pc++] = (byte)token;
    }
    
    void Ce0() {
        if (!err) {
            Ce1();
            do {
                switch (sym) {
                    case PLUS:
                        getsym();
                        Ce1();
                        Gen(PLUS);
                        break;
                    case MINUS:
                        getsym();
                        Ce1();
                        Gen(MINUS);
                        break;
                    default:
                        break;
                }
            }
            while (sym == PLUS || sym == MINUS);
        }
    }
    
    void Ce1() {
        if (!err) {
            Ce2();
            do {
                switch (sym) {
                    case MULT:
                        getsym();
                        Ce2();
                        Gen(MULT);
                        break;
                    case DIV:
                        getsym();
                        Ce2();
                        Gen(DIV);
                        break;
                }
            }
            while (sym == MULT || sym == DIV);
        }
    }
    
    void Ce2() {
        if (!err) {
            Ce3();
            do {
                if (sym == POWER) {
                    getsym();
                    Ce3();
                    Gen(POWER);
                }
            }
            while (sym == POWER);
        }
    }
    
    void Ce3() {
        if (!err) {
            switch (sym) {
                case OPAREN:
                    getsym();
                    Ce0();
                    getsym();
                    break;
                case NUMBER:
                    Gen(PUSHC, nval);
                    getsym();
                    break;
                case IDENT_i:
                    Gen(PUSHI);
                    getsym();
                    break;
                case IDENT_z:
                    Gen(PUSHZ);
                    getsym();
                    break;
                case MINUS:
                    getsym();
                    Ce3();
                    Gen(NEG);
                    break;
                case PLUS:
                    getsym();
                    Ce3();
                    break;
                    
                case FSIN:
                case FCOS:
                case FTAN:
                case FASIN:
                case FACOS:
                case FATAN:
                case FEXP:
                case FINT:
                case FABS:
                case FLOG:
                case FLOG10:
                case FSQRT: {
                    int tsym = sym;
                    getsym();
                    Ce3();
                    Gen(tsym);
                } break;
                    
                case FC:
                    getsym();
                    getsym();
                    Ce3();
                    getsym();
                    Ce3();
                    getsym();
                    Gen(FC);
                    break;
                    
                case SPI:
                    getsym();
                    Gen(PUSHC, (float)M_PI);
                    break;
                case SPHI:
                    getsym();
                    Gen(PUSHC, PHI);
                    break;
                    
                case SNULL:
                    break;
                default:
                    error("unknown symbol: " + ident);
                    break; // syntax error
            }
        }
    }
    
};
#endif
