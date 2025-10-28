\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tubaf-report}[2024-04-18 TUBAF class for writing a report]

\DeclareKeys[tubaf-report]{%
    footmark.code=\PassOptionsToPackage{\CurrentOption}{tubaf-marks},
    footmark.usage=load,
    title-black.if=@tubaf@report@title@black,
    title-black.usage=load,
    title-trapezoid.if=@tubaf@report@title@trapezoid,
    title-trapezoid.usage=load,
}

\SetKeys[tubaf-report]{
    footmark=author-title,
    title-black=false,
    title-trapezoid=true,
}

\DeclareUnknownKeyHandler[tubaf-report]{%
    \PassOptionsToClass{\CurrentOption}{scrreprt}
}

\ProcessKeyOptions[tubaf-report]

\LoadClass[parskip=half]{scrreprt}

\if@tubaf@report@title@black
\PassOptionsToPackage{black=true}{tubaf-title}
\else
\PassOptionsToPackage{black=false}{tubaf-title}
\fi

\if@tubaf@report@title@trapezoid
\PassOptionsToPackage{trapezoid=true}{tubaf-title}
\else
\PassOptionsToPackage{trapezoid=false}{tubaf-title}
\fi

\RequirePackage{tubaf-logo}

\RequirePackage{tubaf-title}

\RequirePackage{tubaf-marks}
