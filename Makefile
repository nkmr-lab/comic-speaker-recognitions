dataset:
	python formatting.py ${data}

calc:
	@python formatting.py ${data}
	@for name in $$(echo ${target} | tr "," " "); do\
		echo start calc $$name && python calc_$${name}.py ${data} && echo finished calc $$name;\
	done

predict:
	@python formatting.py ${data}
	@python predict.py ${data} ${target}
	@python evaluate.py ${data} ${target}

clear:
	@rm -rf data/dataset/*
	@rm -rf data/scores/*
	@rm -rf data/predict/*
