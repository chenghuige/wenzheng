#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   test-tokenize.py
#        \author   chenghuige  
#          \date   2018-02-14 23:49:07.815632
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tokenizer 
import gezi

def tokenize(text):
  print(text)
  results = tokenizer.tokenize(text, True)
  print(results)
  #results = tokenizer.full_tokenize(text)
  #print(results)
  #print(list(zip(results.tokens, results.poses, results.tags, results.ners)))
  #print(len(results))

tokenize('motherfuck')

# TODO fuck with * edit distance ?
tokenize('fu*k')
tokenize('fc*k')
tokenize('fuck')
tokenize('fu**ck')
tokenize('f**in')
tokenize('f*cked')

tokenize('mother\n\nfuck')
tokenize('SECURITYFUCK')
tokenize('WikiNazis')
tokenize('WikiNazis.')
tokenize('Small|penis/////Small|penis/////')

tokenize('Shut up, sit down.  Nig(g)er')
tokenize('you\'re so idi.o.t, so st.u.p.id guy, fu.c.k.i.n.g portuguese, i ha.te the portugueses')

tokenize('''\nOf course a true Dub will say 'depaaarted' or, more likely ""f**in' gone"".''')

tokenize('F u c k i n g   c u n t s get a life you sad f u c k i n g gits.')

tokenize('f u c k  y o u  a s  u  a r e  a  b i t c h  a s s  n i g g e r')

tokenize('T w a t, are you f u c k i n g having a laugh, what kind of thick s h i t, thinks that william shakespeare, my ancestor!, died in. 1616, R.I.P bro,i find that highly rude that you are criticising my ancetors death date......')

tokenize('{{unblock|Please unblock me, my friend was drunk and f*cked up several articles. I would like this account back. Thank you.')

tokenize('chtippDick \n\nYou. You boy. Youuuuuu. Terisoaio! Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah.')

#tokenize('''"== I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f  == 
#I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! == August 2010 == I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! == August 2010 == I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel it needs to be corrected. Have a nice day and f u c k o f f ! ! I will edit whatever I want if I feel"
#''')
#
#tokenize('''Request to Unblock
#I began writing the following statement (excepting its two postscripts) before I was blocked. I suspect that User: AuburnPilot had me blocked, in order to prevent me from complaining to administrators about his misconduct. Although I was directed to request an unblock of User: Physicq210, who had blocked me, requesting such of the latter would require that I censor myself, and write as though User: Physicq210 had acted in good faith, and was one of the good guys. That would be surreal, and would give User: Physicq210 a moral legitimacy belied by his actions. And so, the following request for unblocking is directed at any administrator BUT User: Physicq210, but since I’m not a devious sort, I have nothing against the latter reading this. Indeed, I want User: Physicq210 to read it. For if I can be banned from Wikipedia for standing up to censors, then it is vastly preferable to go down fighting, than to go down the long dark road of self-abnegation and self-censorship, including the use of bad English. After all, if I wanted to be a lackey, I could go to graduate school.
#
#WIKI Administrators’ Notice Board Incidents
#http://en.wikipedia.org/wiki/Wikipedia:Administrators%27_noticeboard/Incidents
#
#User: AuburnPilot
#
#A few days ago, User: AuburnPilot began stalking me, checking wherever I had made edits, and going to the articles in question and vandalizing them. He or she has been manipulating Wikipedia rules as cover for what is clearly political censorship. User: AuburnPilot cannot even stand for Wikipedia readers to find out about factual material that upsets his political applecart through footnotes. Zero tolerance!
#
#He claims of any source he politically dislikes, that it is either “spam” or a “blog” (or “POV,” another favorite dodge for those at Wikipedia seeking to censor those who fail to conform to their own POV), even though the one source is a previously published, classic magazine article otherwise unavailable on the Web (http://geocities.com/nstix/waronpolice.html), which the author has seen fit to publish on his Web site, and the other source is the longest, most thorough exposé yet published on the Duke rape hoax (http://vdare.com/stix/070113_duke.htm).
#
#Note that I am not even talking about censoring writing within articles, since I hadn’t done any writing on the articles in question: Crystal Gail Mangum, Michael Nifong, 2006 Duke University lacrosse team scandal and Racial profiling.
#I’ve been involved in edit wars before, though I have never been the aggressor, and have never initiated an elective edit war, though given the self-assurance that aggressors such as User: AuburnPilot exude, perhaps I ought to reconsider that position. It seems that aggressors rule here.
#
#The reason I am making a formal complaint is that User: AuburnPilot has tonight upped the ante, threatening to have me banned
#(http://en.wikipedia.org/w/index.php?title=User_talk:70.23.199.239&redirect;=no), if I do not surrender to his censorship. Should you find for me, please serve this individual with the wikiquivalent of a cease-and-desist order. Should you, however, find for the censor, please provide an Index of banned publications and a list of official wikicensors.
#
#P.S. January 19, 2007 115A EST. Since beginning this complaint, I see that User: AuburnPilot has in fact succeeded at getting a crony, User: Physicq210, to block me, thus not only getting administrative support in censoring me, but preventing me from responding to his thuggery. (And no, I am not going to use cutesy, euphemistic language. If I were into such deception, I would have become a liar, er, lawyer, and would be worthless as an encyclopedist.) And I was unable to e-mail User: Physicq210 because I am not logged in and “You must be logged in and have a valid authenticated e-mail address in your preferences to send e-mail to other users.” (http://en.wikipedia.org/wiki/Special:Emailuser/Physicq210)
#
#If this isn’t cyber-Stalinism, I don’t know what is! So, let me get this straight. Stalking and censoring an editor while using the equivalent of smiley faces (User: AuburnPilot’s penchant for saying “Thank you” after vandalizing one’s links) is “civil,” but complaining about such abuse counts as “spam, disruption, incivility, and personal attacks.” If that is verily so, then 2 + 2 = 5.
#
#
#P.P.S The message here says that I was “temporarily blocked from editing Wikipedia as a result of your disruptive edits.” What the heck is that supposed to mean? For restoring footnotes and links that User: AuburnPilot had vandalized? If that constitutes “disruptive edits,” then you’ve utterly given up on honest English and all human decency. Sorry, but I don’t have any sympathy for the Devil.''')

tokenize('''I don't want''')
tokenize('''I DON'T want''')
tokenize('''i'm''')
tokenize('''I'm''')
tokenize('''I'M''')
tokenize('''You're bitch''')

print(tokenizer.has('''DONT'T'''))
print(tokenizer.has('''don't'''))
print(tokenizer.has('''i'm'''))
print(tokenizer.has('''I'm'''))
print(tokenizer.has('''I'M'''))

tokenize('he INSISTS upon keeping Boston(as the center of the world.  Clearly biased) in no matter what')
print(gezi.segment.tokenize_filter_empty('he INSISTS upon keeping Boston(as the center of the world.  Clearly biased) in no matter what'))

tokenize('''19 August 2011 (UTC)\nLook who is talking. Legilas is the user who's got Autopatrolled rights although he has  16 articles on the list.''')
tokenize('Hi\n\nwhy did you give User:Barneca a kindness star.')
tokenize('proud to be indonesial')

tokenize('SECURITYFUCK dimension dimenison really fuck you')
tokenize('Uh I hate to break it to you but Jusdafax is not an administrator')
tokenize('I love beijing university. Mike is a Chinese boy. You are a ditry Frenchman')
tokenize('Apple is looking at buying U.K. startup for $1 billion')
